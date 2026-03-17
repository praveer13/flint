//! KV-cache block allocator with reference counting for copy-on-write.
//!
//! In LLM inference, the model maintains a Key-Value cache — the attention state
//! for every token processed so far. For large models (e.g., 70B parameters),
//! each token's KV-cache can be ~1.2 MB, so a 4096-token sequence consumes ~5 GB.
//! PagedAttention (from vLLM) manages this like OS virtual memory: the KV-cache
//! is divided into fixed-size *blocks* (e.g., 16 tokens per block), and a block
//! table maps `(sequence_id, logical_block) -> physical GPU memory address`.
//!
//! This allocator owns the free list of physical block IDs and the per-block
//! reference counts. It does NOT touch GPU memory — it only tracks metadata.
//! The actual GPU memory is allocated once at startup by the vLLM worker and
//! addressed by block ID.
//!
//! ## Free-stack pattern
//!
//! Free block IDs are stored in a stack (array + top-of-stack index). `alloc`
//! pops from the stack; `free` pushes back. Both are O(1) with no branching
//! beyond the bounds check. We use a stack rather than a linked list because:
//!
//!   - **Cache locality:** The stack is a contiguous array. The hot end (near
//!     `free_top`) stays in L1. A linked list would scatter block metadata
//!     across cache lines, causing misses on every alloc/free.
//!   - **No pointer chasing:** The scheduler calls alloc/free hundreds of times
//!     per iteration (every 10-50 ms). Pointer chasing through a linked list
//!     would add ~50-100 ns per operation on a cache miss. The stack keeps us
//!     under 10 ns.
//!   - **Predictable memory:** The stack is pre-allocated at startup. No dynamic
//!     allocation on the hot path.
//!
//! ## Copy-on-write (CoW)
//!
//! When multiple sequences share a common prefix (e.g., a system prompt), they
//! can reference the same physical KV-cache blocks instead of duplicating them.
//! This is managed via per-block reference counts:
//!
//!   - `incref`: A new sequence shares a block -> ref_count += 1.
//!   - `free`: A sequence releases a block -> ref_count -= 1. The block returns
//!     to the free pool only when ref_count reaches 0.
//!   - `cow`: A sequence needs to *modify* a shared block (e.g., append new KV
//!     entries after a shared prefix). If ref_count == 1, it's the sole owner
//!     and can modify in place. If ref_count > 1, we allocate a fresh block for
//!     the caller and decrement the shared block's ref_count. The caller then
//!     copies the KV data to the new block (GPU-side memcpy, not our concern).
//!
//! A typical scenario: 100 chat requests all start with the same 2048-token
//! system prompt. That's 128 blocks (at 16 tokens/block). Without CoW, we'd
//! need 128 * 100 = 12,800 blocks just for system prompts. With CoW, we need
//! 128 blocks total, each with ref_count = 100. When a request diverges from
//! the shared prefix, `cow` gives it a private copy of just the divergent block.

const std = @import("std");

/// A free-list stack allocator for KV-cache memory blocks with reference
/// counting for copy-on-write.
///
/// This struct does not own the backing memory for the free stack or reference
/// count arrays — the caller provides pre-allocated slices. This makes the
/// allocator embeddable in shared memory or any fixed-size arena without
/// requiring a heap allocator.
///
/// All operations are O(1). None of them allocate or do I/O. The allocator
/// is single-threaded — the scheduler is the sole caller and runs on a
/// dedicated OS thread.
pub const BlockAllocator = struct {
    /// Stack of free block IDs. Block IDs are pushed/popped from the end
    /// indexed by `free_top`. Elements at indices `[0, free_top)` are the
    /// currently available blocks.
    free_stack: []u32,

    /// Index of the next free slot in `free_stack`. 0 means empty (no blocks
    /// available), `total_blocks` means all blocks are free.
    free_top: u32,

    /// Per-block reference count. A block with ref_count 0 is free (on the
    /// stack). A block with ref_count 1 is exclusively owned by one sequence.
    /// A block with ref_count > 1 is shared via copy-on-write.
    ref_counts: []u16,

    /// Total number of blocks managed by this allocator.
    total_blocks: u32,

    /// Initialize a block allocator over pre-allocated buffers.
    ///
    /// - `total_blocks`: Number of blocks to manage. Must be <= the length of
    ///   both `free_stack_buf` and `ref_count_buf`.
    /// - `free_stack_buf`: Backing storage for the free stack. Must have at
    ///   least `total_blocks` elements.
    /// - `ref_count_buf`: Backing storage for reference counts. Must have at
    ///   least `total_blocks` elements.
    ///
    /// After init, all blocks are free and `available() == total_blocks`.
    pub fn init(total_blocks: u32, free_stack_buf: []u32, ref_count_buf: []u16) BlockAllocator {
        std.debug.assert(free_stack_buf.len >= total_blocks);
        std.debug.assert(ref_count_buf.len >= total_blocks);

        // Fill free stack with block IDs 0, 1, 2, ..., total_blocks - 1.
        for (0..total_blocks) |i| {
            free_stack_buf[i] = @intCast(i);
        }

        // All ref counts start at 0 (free).
        for (0..total_blocks) |i| {
            ref_count_buf[i] = 0;
        }

        return .{
            .free_stack = free_stack_buf[0..total_blocks],
            .free_top = total_blocks,
            .ref_counts = ref_count_buf[0..total_blocks],
            .total_blocks = total_blocks,
        };
    }

    /// Allocate one block. Returns the block ID, or `null` if no blocks are
    /// available.
    ///
    /// The returned block has a reference count of 1 (exclusively owned by
    /// the caller). O(1) — pops from the free stack.
    pub fn alloc(self: *BlockAllocator) ?u32 {
        if (self.free_top == 0) return null;
        self.free_top -= 1;
        const block = self.free_stack[self.free_top];
        self.ref_counts[block] = 1;
        return block;
    }

    /// Free one block. Decrements the reference count; the block returns to
    /// the free stack only when the count reaches 0.
    ///
    /// It is the caller's responsibility to ensure `block` is a valid block ID
    /// with a non-zero reference count. Freeing an already-free block is a
    /// programming error.
    pub fn free(self: *BlockAllocator, block: u32) void {
        std.debug.assert(block < self.total_blocks);
        std.debug.assert(self.ref_counts[block] > 0);

        self.ref_counts[block] -= 1;
        if (self.ref_counts[block] == 0) {
            self.free_stack[self.free_top] = block;
            self.free_top += 1;
        }
    }

    /// Copy-on-write: prepare a block for exclusive modification.
    ///
    /// If the block's reference count is 1 (sole owner), returns the same
    /// block ID — the caller can modify it in place without copying.
    ///
    /// If the reference count is > 1 (shared), allocates a new block,
    /// decrements the old block's reference count, and returns the new block
    /// ID. The caller is responsible for copying the KV-cache data from the
    /// old block to the new one (GPU-side memcpy).
    ///
    /// Returns `null` if the block is shared and no free blocks are available
    /// for the copy.
    pub fn cow(self: *BlockAllocator, block: u32) ?u32 {
        std.debug.assert(block < self.total_blocks);
        std.debug.assert(self.ref_counts[block] > 0);

        if (self.ref_counts[block] == 1) return block; // Sole owner.
        const new_block = self.alloc() orelse return null;
        self.ref_counts[block] -= 1;
        return new_block;
    }

    /// Increment the reference count for a block (share it with another
    /// sequence).
    ///
    /// The block must already be allocated (ref_count >= 1). This is used when
    /// a new sequence wants to reuse an existing block's KV-cache data, e.g.,
    /// when two requests share a common system prompt prefix.
    pub fn incref(self: *BlockAllocator, block: u32) void {
        std.debug.assert(block < self.total_blocks);
        std.debug.assert(self.ref_counts[block] > 0);
        self.ref_counts[block] += 1;
    }

    /// Number of free blocks currently available for allocation.
    pub fn available(self: *const BlockAllocator) u32 {
        return self.free_top;
    }

    /// Get the current reference count for a block.
    ///
    /// A ref_count of 0 means the block is free. 1 means exclusively owned.
    /// > 1 means shared via copy-on-write.
    pub fn refCount(self: *const BlockAllocator, block: u32) u16 {
        std.debug.assert(block < self.total_blocks);
        return self.ref_counts[block];
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Helper: create a BlockAllocator backed by test-allocated slices.
fn makeTestAllocator(ally: std.mem.Allocator, total_blocks: u32) !struct {
    block_alloc: BlockAllocator,
    free_stack_buf: []u32,
    ref_count_buf: []u16,
} {
    const free_stack_buf = try ally.alloc(u32, total_blocks);
    const ref_count_buf = try ally.alloc(u16, total_blocks);
    const block_alloc = BlockAllocator.init(total_blocks, free_stack_buf, ref_count_buf);
    return .{
        .block_alloc = block_alloc,
        .free_stack_buf = free_stack_buf,
        .ref_count_buf = ref_count_buf,
    };
}

fn freeTestAllocator(ally: std.mem.Allocator, ctx: anytype) void {
    ally.free(ctx.free_stack_buf);
    ally.free(ctx.ref_count_buf);
}

test "init state — all blocks free" {
    var ctx = try makeTestAllocator(testing.allocator, 64);
    defer freeTestAllocator(testing.allocator, ctx);

    try testing.expectEqual(@as(u32, 64), ctx.block_alloc.available());

    // All ref counts should be 0.
    for (0..64) |i| {
        try testing.expectEqual(@as(u16, 0), ctx.block_alloc.refCount(@intCast(i)));
    }
}

test "alloc/free cycle — single block" {
    var ctx = try makeTestAllocator(testing.allocator, 16);
    defer freeTestAllocator(testing.allocator, ctx);

    const block = ctx.block_alloc.alloc().?;
    try testing.expectEqual(@as(u16, 1), ctx.block_alloc.refCount(block));
    try testing.expectEqual(@as(u32, 15), ctx.block_alloc.available());

    ctx.block_alloc.free(block);
    try testing.expectEqual(@as(u16, 0), ctx.block_alloc.refCount(block));
    try testing.expectEqual(@as(u32, 16), ctx.block_alloc.available());
}

test "alloc until exhausted — returns null" {
    var ctx = try makeTestAllocator(testing.allocator, 8);
    defer freeTestAllocator(testing.allocator, ctx);

    // Allocate all 8 blocks.
    for (0..8) |_| {
        try testing.expect(ctx.block_alloc.alloc() != null);
    }

    try testing.expectEqual(@as(u32, 0), ctx.block_alloc.available());
    try testing.expectEqual(@as(?u32, null), ctx.block_alloc.alloc());
}

test "free restores availability — alloc all, free one, alloc succeeds" {
    var ctx = try makeTestAllocator(testing.allocator, 4);
    defer freeTestAllocator(testing.allocator, ctx);

    // Alloc all blocks.
    var blocks: [4]u32 = undefined;
    for (0..4) |i| {
        blocks[i] = ctx.block_alloc.alloc().?;
    }
    try testing.expectEqual(@as(?u32, null), ctx.block_alloc.alloc());

    // Free one block.
    ctx.block_alloc.free(blocks[2]);
    try testing.expectEqual(@as(u32, 1), ctx.block_alloc.available());

    // Alloc again should succeed and return the freed block.
    const new_block = ctx.block_alloc.alloc().?;
    try testing.expectEqual(blocks[2], new_block);
    try testing.expectEqual(@as(u32, 0), ctx.block_alloc.available());
}

test "cow sole owner — returns same block, no new allocation" {
    var ctx = try makeTestAllocator(testing.allocator, 8);
    defer freeTestAllocator(testing.allocator, ctx);

    const block = ctx.block_alloc.alloc().?;
    const before = ctx.block_alloc.available();

    // CoW on a block with ref_count == 1 should return the same block.
    const cow_block = ctx.block_alloc.cow(block).?;
    try testing.expectEqual(block, cow_block);
    try testing.expectEqual(@as(u16, 1), ctx.block_alloc.refCount(block));

    // No blocks consumed by the cow operation.
    try testing.expectEqual(before, ctx.block_alloc.available());
}

test "cow shared block — returns new block, decrements old ref" {
    var ctx = try makeTestAllocator(testing.allocator, 8);
    defer freeTestAllocator(testing.allocator, ctx);

    const block = ctx.block_alloc.alloc().?;
    ctx.block_alloc.incref(block);
    try testing.expectEqual(@as(u16, 2), ctx.block_alloc.refCount(block));

    const before = ctx.block_alloc.available();
    const new_block = ctx.block_alloc.cow(block).?;

    // Must be a different block.
    try testing.expect(new_block != block);
    // Old block's ref count decremented from 2 to 1.
    try testing.expectEqual(@as(u16, 1), ctx.block_alloc.refCount(block));
    // New block has ref count 1 (exclusively owned).
    try testing.expectEqual(@as(u16, 1), ctx.block_alloc.refCount(new_block));
    // One block consumed from the free pool.
    try testing.expectEqual(before - 1, ctx.block_alloc.available());
}

test "cow when full — returns null" {
    var ctx = try makeTestAllocator(testing.allocator, 4);
    defer freeTestAllocator(testing.allocator, ctx);

    // Alloc all blocks.
    const shared = ctx.block_alloc.alloc().?;
    ctx.block_alloc.incref(shared); // ref_count = 2
    for (0..3) |_| {
        _ = ctx.block_alloc.alloc();
    }

    try testing.expectEqual(@as(u32, 0), ctx.block_alloc.available());
    try testing.expectEqual(@as(u16, 2), ctx.block_alloc.refCount(shared));

    // CoW on a shared block when no free blocks are available should return null.
    try testing.expectEqual(@as(?u32, null), ctx.block_alloc.cow(shared));

    // Original block's ref count should be unchanged (cow failed, no side effects).
    try testing.expectEqual(@as(u16, 2), ctx.block_alloc.refCount(shared));
}

test "incref/free with sharing — block freed only after last reference" {
    var ctx = try makeTestAllocator(testing.allocator, 8);
    defer freeTestAllocator(testing.allocator, ctx);

    const block = ctx.block_alloc.alloc().?;
    // Share with 2 more sequences: ref_count goes 1 -> 2 -> 3.
    ctx.block_alloc.incref(block);
    ctx.block_alloc.incref(block);
    try testing.expectEqual(@as(u16, 3), ctx.block_alloc.refCount(block));

    const free_before = ctx.block_alloc.available();

    // First free: ref_count 3 -> 2. Block stays allocated.
    ctx.block_alloc.free(block);
    try testing.expectEqual(@as(u16, 2), ctx.block_alloc.refCount(block));
    try testing.expectEqual(free_before, ctx.block_alloc.available());

    // Second free: ref_count 2 -> 1. Block stays allocated.
    ctx.block_alloc.free(block);
    try testing.expectEqual(@as(u16, 1), ctx.block_alloc.refCount(block));
    try testing.expectEqual(free_before, ctx.block_alloc.available());

    // Third free: ref_count 1 -> 0. Block returns to free pool.
    ctx.block_alloc.free(block);
    try testing.expectEqual(@as(u16, 0), ctx.block_alloc.refCount(block));
    try testing.expectEqual(free_before + 1, ctx.block_alloc.available());
}

test "alloc returns distinct block IDs" {
    var ctx = try makeTestAllocator(testing.allocator, 16);
    defer freeTestAllocator(testing.allocator, ctx);

    var seen = [_]bool{false} ** 16;
    for (0..16) |_| {
        const block = ctx.block_alloc.alloc().?;
        try testing.expect(!seen[block]);
        seen[block] = true;
    }

    // All 16 blocks should have been returned.
    for (seen) |s| {
        try testing.expect(s);
    }
}
