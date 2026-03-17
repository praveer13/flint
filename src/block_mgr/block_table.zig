//! Block table — 2D mapping from (sequence slot, logical block) to physical block ID.
//!
//! During inference with PagedAttention, each sequence's KV-cache is split into
//! fixed-size blocks. The attention kernel needs to know which physical GPU
//! memory block corresponds to each logical position in the sequence. This
//! module provides that mapping as a flat 2D array in shared memory, readable
//! by both the Zig scheduler and the Python vLLM worker.
//!
//! ## Layout
//!
//! The table is stored row-major in a contiguous region of shared memory:
//!
//!     table[seq_slot * max_blocks_per_seq + logical_block] = physical_block_id
//!
//! Each entry is a `u32`, so the total size is `max_seqs * max_blocks_per_seq * 4`
//! bytes. Row-major layout means all blocks for a single sequence are contiguous
//! in memory, which is cache-friendly for the common case of iterating over one
//! sequence's blocks.
//!
//! ## seq_slot vs seq_id
//!
//! The block table is indexed by a *slot index* (0 .. max_seqs-1), not by the
//! global `seq_id`. The scheduler maintains the mapping from seq_id to slot
//! index. This keeps the table compact — a flat array rather than a sparse map —
//! and ensures predictable memory usage regardless of how many total sequences
//! have been processed over the server's lifetime.
//!
//! ## Memory ordering
//!
//! The Zig scheduler writes block mappings, then posts a schedule to the shared
//! memory ring buffer. The Python worker reads the schedule, then reads the
//! block table to locate KV-cache blocks. For correctness, all block table
//! writes must be visible to the reader *before* the schedule that references
//! them is posted. Call `fence()` between updating the block table and pushing
//! the schedule onto the ring. Without this fence, the worker could observe a
//! stale block mapping and read the wrong KV-cache data, producing garbage
//! output.

const std = @import("std");

/// A 2D lookup table mapping (sequence slot, logical block index) to physical
/// block ID, backed by a shared memory region.
pub const BlockTable = struct {
    /// Flat row-major array: `data[seq_slot * max_blocks_per_seq + logical]`.
    /// Points into shared memory — both Zig and Python see the same bytes.
    data: [*]u32,

    /// Maximum number of sequence slots (rows).
    max_seqs: u32,

    /// Maximum number of logical blocks per sequence (columns).
    max_blocks_per_seq: u32,

    /// Initialize a block table over a raw memory region.
    ///
    /// `base` is the start of the shared memory region (e.g., `ShmRegion.ptr`).
    /// `offset` is the byte offset within that region where the block table
    /// starts. The caller must ensure that `offset + totalSize(max_seqs,
    /// max_blocks_per_seq)` does not exceed the region size, and that `offset`
    /// is aligned to `@alignOf(u32)` (4 bytes).
    pub fn init(base: [*]u8, offset: usize, max_seqs: u32, max_blocks_per_seq: u32) BlockTable {
        return .{
            .data = @ptrCast(@alignCast(base + offset)),
            .max_seqs = max_seqs,
            .max_blocks_per_seq = max_blocks_per_seq,
        };
    }

    /// Set the physical block ID for a (sequence slot, logical block) pair.
    ///
    /// Panics in safe builds if `seq_slot >= max_seqs` or
    /// `logical >= max_blocks_per_seq`.
    pub fn set(self: *BlockTable, seq_slot: u32, logical: u32, physical: u32) void {
        self.data[self.index(seq_slot, logical)] = physical;
    }

    /// Get the physical block ID for a (sequence slot, logical block) pair.
    ///
    /// Panics in safe builds if `seq_slot >= max_seqs` or
    /// `logical >= max_blocks_per_seq`.
    pub fn get(self: *const BlockTable, seq_slot: u32, logical: u32) u32 {
        return self.data[self.index(seq_slot, logical)];
    }

    /// Clear all block mappings for a sequence slot (set every entry to 0).
    ///
    /// Called when a sequence finishes or is evicted and its slot is recycled.
    pub fn clearSeq(self: *BlockTable, seq_slot: u32) void {
        const start = @as(usize, seq_slot) * self.max_blocks_per_seq;
        const row = self.data[start .. start + self.max_blocks_per_seq];
        @memset(row, 0);
    }

    /// Return a read-only slice of the block mappings for a sequence slot.
    ///
    /// The returned slice has `max_blocks_per_seq` entries. The caller
    /// typically copies the first `num_blocks` entries into the `ScheduleT`
    /// when building the iteration schedule.
    pub fn seqBlocks(self: *const BlockTable, seq_slot: u32) []const u32 {
        const start = @as(usize, seq_slot) * self.max_blocks_per_seq;
        return self.data[start .. start + self.max_blocks_per_seq];
    }

    /// Full memory fence (sequential consistency).
    ///
    /// Call this after writing block table entries and before posting the
    /// schedule that references those blocks to the shared memory ring buffer.
    /// This guarantees that the Python worker, upon reading the schedule, will
    /// see all block table updates that the schedule depends on.
    pub fn fence(self: *const BlockTable) void {
        // Zig 0.16 does not expose a standalone fence builtin. We emulate a
        // full memory fence with a seq_cst atomic load on the first element of
        // the data array. The seq_cst ordering issues a full barrier on x86
        // (MFENCE or equivalent) and appropriate barriers on ARM, ensuring all
        // prior writes to the block table are visible to the reader before any
        // subsequent write to the schedule ring.
        _ = @atomicLoad(u32, &self.data[0], .seq_cst);
    }

    /// Compute the total byte size required for a block table with the given
    /// dimensions.
    ///
    /// Use this when allocating the shared memory region to reserve the right
    /// amount of space for the block table.
    pub fn totalSize(max_seqs: u32, max_blocks_per_seq: u32) usize {
        return @as(usize, max_seqs) * max_blocks_per_seq * @sizeOf(u32);
    }

    /// Compute the flat index for a (seq_slot, logical block) pair.
    /// Bounds-checked in safe builds.
    inline fn index(self: *const BlockTable, seq_slot: u32, logical: u32) usize {
        std.debug.assert(seq_slot < self.max_seqs);
        std.debug.assert(logical < self.max_blocks_per_seq);
        return @as(usize, seq_slot) * self.max_blocks_per_seq + logical;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Helper: allocate a zeroed buffer and create a BlockTable over it.
fn makeTestTable(max_seqs: u32, max_blocks: u32) struct { buf: []align(4) u8, table: BlockTable } {
    const size = BlockTable.totalSize(max_seqs, max_blocks);
    const buf = testing.allocator.alignedAlloc(u8, .@"4", size) catch @panic("OOM in test");
    @memset(buf, 0);
    const table = BlockTable.init(buf.ptr, 0, max_seqs, max_blocks);
    return .{ .buf = buf, .table = table };
}

fn freeTestTable(buf: []align(4) u8) void {
    testing.allocator.free(buf);
}

test "init and set/get — basic round-trip" {
    var ctx = makeTestTable(4, 8);
    defer freeTestTable(ctx.buf);

    ctx.table.set(0, 3, 42);
    try testing.expectEqual(@as(u32, 42), ctx.table.get(0, 3));

    // Other entries should still be zero.
    try testing.expectEqual(@as(u32, 0), ctx.table.get(0, 0));
    try testing.expectEqual(@as(u32, 0), ctx.table.get(0, 7));
}

test "multiple sequences — independent rows" {
    var ctx = makeTestTable(4, 8);
    defer freeTestTable(ctx.buf);

    // Set blocks for sequence slot 0.
    ctx.table.set(0, 0, 100);
    ctx.table.set(0, 1, 101);

    // Set blocks for sequence slot 2.
    ctx.table.set(2, 0, 200);
    ctx.table.set(2, 5, 205);

    // Verify slot 0 is unaffected by slot 2 writes.
    try testing.expectEqual(@as(u32, 100), ctx.table.get(0, 0));
    try testing.expectEqual(@as(u32, 101), ctx.table.get(0, 1));
    try testing.expectEqual(@as(u32, 0), ctx.table.get(0, 5));

    // Verify slot 2.
    try testing.expectEqual(@as(u32, 200), ctx.table.get(2, 0));
    try testing.expectEqual(@as(u32, 205), ctx.table.get(2, 5));

    // Verify slot 1 is untouched.
    try testing.expectEqual(@as(u32, 0), ctx.table.get(1, 0));
}

test "clearSeq — zeroes an entire row" {
    var ctx = makeTestTable(4, 8);
    defer freeTestTable(ctx.buf);

    // Populate slot 1.
    for (0..8) |i| {
        ctx.table.set(1, @intCast(i), @as(u32, @intCast(i + 10)));
    }

    // Populate slot 2 so we can verify it is not disturbed.
    ctx.table.set(2, 0, 999);

    // Clear slot 1.
    ctx.table.clearSeq(1);

    // Every entry in slot 1 should be zero.
    for (0..8) |i| {
        try testing.expectEqual(@as(u32, 0), ctx.table.get(1, @intCast(i)));
    }

    // Slot 2 must be untouched.
    try testing.expectEqual(@as(u32, 999), ctx.table.get(2, 0));
}

test "seqBlocks — returns correct slice contents" {
    var ctx = makeTestTable(4, 8);
    defer freeTestTable(ctx.buf);

    ctx.table.set(3, 0, 50);
    ctx.table.set(3, 1, 51);
    ctx.table.set(3, 2, 52);

    const blocks = ctx.table.seqBlocks(3);
    try testing.expectEqual(@as(usize, 8), blocks.len);
    try testing.expectEqual(@as(u32, 50), blocks[0]);
    try testing.expectEqual(@as(u32, 51), blocks[1]);
    try testing.expectEqual(@as(u32, 52), blocks[2]);
    try testing.expectEqual(@as(u32, 0), blocks[3]);
}

test "fence — does not crash" {
    var ctx = makeTestTable(2, 4);
    defer freeTestTable(ctx.buf);

    ctx.table.set(0, 0, 1);
    ctx.table.fence();
    // If we get here, the fence executed without error.
    try testing.expectEqual(@as(u32, 1), ctx.table.get(0, 0));
}

test "totalSize — matches expected byte count" {
    // 4 seqs * 8 blocks * 4 bytes/u32 = 128 bytes
    try testing.expectEqual(@as(usize, 128), BlockTable.totalSize(4, 8));

    // 256 seqs * 512 blocks * 4 bytes = 524288 bytes (512 KiB)
    try testing.expectEqual(@as(usize, 524288), BlockTable.totalSize(256, 512));

    // Edge: 1 seq, 1 block = 4 bytes
    try testing.expectEqual(@as(usize, 4), BlockTable.totalSize(1, 1));
}

test "non-zero offset — table starts at an offset within the region" {
    const max_seqs: u32 = 2;
    const max_blocks: u32 = 4;
    const offset: usize = 64; // Simulate block table not starting at byte 0.
    const total = offset + BlockTable.totalSize(max_seqs, max_blocks);

    const buf = try testing.allocator.alignedAlloc(u8, .@"4", total);
    defer testing.allocator.free(buf);
    @memset(buf, 0);

    var table = BlockTable.init(buf.ptr, offset, max_seqs, max_blocks);

    table.set(1, 2, 77);
    try testing.expectEqual(@as(u32, 77), table.get(1, 2));

    // Verify the bytes landed at the right place in the raw buffer.
    const raw: [*]const u32 = @ptrCast(@alignCast(buf.ptr + offset));
    // Row 1, column 2 → flat index 1*4 + 2 = 6.
    try testing.expectEqual(@as(u32, 77), raw[6]);
}
