//! Lock-free Single-Producer Single-Consumer (SPSC) ring buffer for shared
//! memory IPC.
//!
//! This is the core communication primitive between the Zig server process and
//! the Python GPU worker processes. Two instances are used in practice:
//!
//!   - **Schedule ring:** Zig (producer) writes iteration schedules, Python
//!     (consumer) reads them and runs GPU forward passes.
//!   - **Completion ring:** Python (producer) writes output tokens, Zig
//!     (consumer) reads them and streams to clients via SSE.
//!
//! ## Why lock-free?
//!
//! Locks across process boundaries are painful (robust mutexes, recovery on
//! crash, priority inversion). With SPSC, we only need atomic loads and stores
//! on two counters — `head` (consumer) and `tail` (producer). No CAS, no
//! futex, no kernel involvement on the data path.
//!
//! ## Memory ordering rationale
//!
//! The ring uses acquire/release semantics to ensure correctness without
//! sequential consistency (which would be unnecessarily expensive):
//!
//!   - **Producer (`tryPush`):** Writes slot data first, then does a *release*
//!     store to `tail`. The release ordering guarantees that the slot write is
//!     visible to any thread that *acquires* the new tail value. In plain
//!     English: "the data is fully written before I announce it's available."
//!
//!   - **Consumer (`tryPop`):** Does an *acquire* load of `tail` to see how
//!     far the producer has written. The acquire ordering guarantees that all
//!     writes made by the producer before its release store are visible. In
//!     plain English: "once I see a new tail, I'm guaranteed to see the data."
//!
//!   - **Own counter:** Each side reads its own counter with *monotonic*
//!     ordering because it is the sole writer — the value cannot change under
//!     it, so no synchronization is needed.
//!
//! ## Cache-line padding
//!
//! `head` and `tail` live on separate 64-byte cache lines. Without this
//! separation, the producer's stores to `tail` would invalidate the consumer's
//! cache line containing `head` (and vice versa) — a phenomenon called *false
//! sharing*. On modern x86 CPUs, a cache line is 64 bytes, so we pad each
//! counter to exactly 64 bytes.
//!
//! ## Indices are u64 and never wrap
//!
//! `head` and `tail` are monotonically increasing u64 counters. We use
//! `index % capacity` to compute the slot position. This avoids the classic
//! ring buffer ambiguity where head == tail could mean "empty" or "full" — with
//! u64 counters, full is `tail - head >= capacity` and empty is `head == tail`.
//! A u64 counter would take billions of years to overflow at nanosecond
//! granularity, so wraparound is not a concern.

const std = @import("std");

/// A generic SPSC ring buffer parameterized by the element type `T`.
///
/// `T` should be an `extern struct` (for cross-language ABI stability) but the
/// ring itself works with any copyable type. The ring operates on a contiguous
/// byte region — typically a slice of a shared memory mapping — and does not
/// allocate.
pub fn SpscRing(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Ring buffer header laid out in shared memory.
        ///
        /// Each counter occupies its own 64-byte cache line to prevent false
        /// sharing between producer and consumer. The header is an `extern
        /// struct` so its layout is deterministic across Zig and C/Python.
        pub const Header = extern struct {
            /// Consumer position. The consumer (reader) advances this after
            /// copying a slot's data out. Only the consumer writes to `head`;
            /// the producer only reads it (to check if the ring is full).
            head: u64 align(64),
            _pad0: [56]u8 = .{0} ** 56,

            /// Producer position. The producer (writer) advances this after
            /// writing data into a slot. Only the producer writes to `tail`;
            /// the consumer only reads it (to check if new data is available).
            tail: u64 align(64),
            _pad1: [56]u8 = .{0} ** 56,
        };

        /// Pointer to the header in shared memory.
        header: *Header,

        /// Pointer to the first slot. Slots are contiguous in memory
        /// immediately after the header.
        slots: [*]T,

        /// Number of slots. Must be the same value used by both producer and
        /// consumer.
        capacity: u32,

        /// Compute the total byte size needed in shared memory for a ring of
        /// the given capacity.
        ///
        /// This accounts for the header (with cache-line padding) plus
        /// `capacity` slots of type `T`. Use this when allocating shared
        /// memory regions.
        pub fn totalSize(capacity: u32) usize {
            return @sizeOf(Header) + @sizeOf(T) * capacity;
        }

        /// Initialize a ring buffer over a raw byte region.
        ///
        /// - `base`: Start of the shared memory region (e.g., from `mmap`).
        /// - `offset`: Byte offset within `base` where this ring starts.
        /// - `capacity`: Number of slots.
        ///
        /// The caller must ensure that `base[offset..offset + totalSize(capacity)]`
        /// is valid, writable memory, and zero-initialized (so `head` and
        /// `tail` start at 0).
        pub fn init(base: [*]u8, offset: usize, capacity: u32) Self {
            const header_ptr: *Header = @ptrCast(@alignCast(base + offset));
            const slots_ptr: [*]T = @ptrCast(@alignCast(base + offset + @sizeOf(Header)));
            return .{
                .header = header_ptr,
                .slots = slots_ptr,
                .capacity = capacity,
            };
        }

        /// Try to push (produce) one item into the ring.
        ///
        /// Returns `true` if the item was written, `false` if the ring is
        /// full (consumer hasn't caught up). This never blocks.
        ///
        /// **Atomic protocol:**
        ///   1. Read our own `tail` (monotonic — we're the sole writer).
        ///   2. Read the consumer's `head` (acquire — synchronizes with the
        ///      consumer's release store, ensuring we see a current value).
        ///   3. If `tail - head >= capacity`, the ring is full.
        ///   4. Copy `item` into `slots[tail % capacity]`.
        ///   5. Release-store `tail + 1` — this publishes the new data. The
        ///      release ordering ensures the slot write (step 4) is visible
        ///      to any consumer that acquires this new tail value.
        pub fn tryPush(self: *Self, item: *const T) bool {
            const tail = @atomicLoad(u64, &self.header.tail, .monotonic);
            const head = @atomicLoad(u64, &self.header.head, .acquire);
            if (tail - head >= self.capacity) return false;

            self.slots[tail % self.capacity] = item.*;
            @atomicStore(u64, &self.header.tail, tail + 1, .release);
            return true;
        }

        /// Try to pop (consume) one item from the ring.
        ///
        /// Returns `true` if an item was read into `out`, `false` if the ring
        /// is empty. This never blocks.
        ///
        /// **Atomic protocol:**
        ///   1. Read our own `head` (monotonic — we're the sole writer).
        ///   2. Read the producer's `tail` (acquire — synchronizes with the
        ///      producer's release store, ensuring we see the data written
        ///      before the tail was advanced).
        ///   3. If `head == tail`, the ring is empty.
        ///   4. Copy `slots[head % capacity]` into `out`.
        ///   5. Release-store `head + 1` — this frees the slot. The release
        ///      ordering ensures our read (step 4) completes before the
        ///      producer sees the updated head and potentially overwrites
        ///      the slot.
        pub fn tryPop(self: *Self, out: *T) bool {
            const head = @atomicLoad(u64, &self.header.head, .monotonic);
            const tail = @atomicLoad(u64, &self.header.tail, .acquire);
            if (head == tail) return false;

            out.* = self.slots[head % self.capacity];
            @atomicStore(u64, &self.header.head, head + 1, .release);
            return true;
        }

        /// Return the approximate number of items in the ring.
        ///
        /// This is a best-effort snapshot — by the time the caller acts on it,
        /// the producer may have pushed more items or the consumer may have
        /// popped some. Useful for metrics and backpressure heuristics, NOT
        /// for correctness decisions.
        ///
        /// Uses monotonic ordering on both loads because we don't need
        /// happens-before relationships — we just want a rough count.
        /// The saturating subtract (`-|`) guards against the unlikely case
        /// where we read a stale tail < head due to reordering.
        pub fn len(self: *const Self) u32 {
            const tail = @atomicLoad(u64, &self.header.tail, .monotonic);
            const head = @atomicLoad(u64, &self.header.head, .monotonic);
            return @intCast(tail -| head);
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// A simple extern struct to exercise the ring with a non-trivial type.
const TestItem = extern struct {
    seq_id: u64,
    value: u32,
    flag: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
};

/// Allocate a zero-initialized, 64-byte-aligned buffer and return a ring
/// over it. The caller must free the buffer.
fn makeTestRing(comptime T: type, allocator: std.mem.Allocator, capacity: u32) !struct { ring: SpscRing(T), buf: []align(64) u8 } {
    const total = SpscRing(T).totalSize(capacity);
    const buf = try allocator.alignedAlloc(u8, .@"64", total);
    @memset(buf, 0);
    const ring = SpscRing(T).init(buf.ptr, 0, capacity);
    return .{ .ring = ring, .buf = buf };
}

test "basic push and pop — FIFO order preserved" {
    const alloc = testing.allocator;
    var ctx = try makeTestRing(u64, alloc, 8);
    defer alloc.free(ctx.buf);

    // Push 5 values.
    for (0..5) |i| {
        var val: u64 = @intCast(i * 10);
        try testing.expect(ctx.ring.tryPush(&val));
    }

    // Pop them back and verify order.
    for (0..5) |i| {
        var out: u64 = undefined;
        try testing.expect(ctx.ring.tryPop(&out));
        try testing.expectEqual(@as(u64, @intCast(i * 10)), out);
    }
}

test "full ring — tryPush returns false" {
    const alloc = testing.allocator;
    const capacity: u32 = 4;
    var ctx = try makeTestRing(u64, alloc, capacity);
    defer alloc.free(ctx.buf);

    // Fill the ring to capacity.
    for (0..capacity) |i| {
        var val: u64 = @intCast(i);
        try testing.expect(ctx.ring.tryPush(&val));
    }

    // Next push must fail.
    var extra: u64 = 999;
    try testing.expect(!ctx.ring.tryPush(&extra));
}

test "empty ring — tryPop returns false" {
    const alloc = testing.allocator;
    var ctx = try makeTestRing(u64, alloc, 4);
    defer alloc.free(ctx.buf);

    var out: u64 = undefined;
    try testing.expect(!ctx.ring.tryPop(&out));
}

test "wraparound — indices wrap via modulo correctly" {
    const alloc = testing.allocator;
    const capacity: u32 = 4;
    var ctx = try makeTestRing(u64, alloc, capacity);
    defer alloc.free(ctx.buf);

    // First pass: fill and drain.
    for (0..capacity) |i| {
        var val: u64 = @intCast(i);
        try testing.expect(ctx.ring.tryPush(&val));
    }
    for (0..capacity) |i| {
        var out: u64 = undefined;
        try testing.expect(ctx.ring.tryPop(&out));
        try testing.expectEqual(@as(u64, @intCast(i)), out);
    }

    // Second pass: fill and drain again — this exercises the modulo path
    // because head and tail are now >= capacity.
    for (0..capacity) |i| {
        var val: u64 = @intCast(100 + i);
        try testing.expect(ctx.ring.tryPush(&val));
    }
    for (0..capacity) |i| {
        var out: u64 = undefined;
        try testing.expect(ctx.ring.tryPop(&out));
        try testing.expectEqual(@as(u64, @intCast(100 + i)), out);
    }
}

test "len — reflects pushes and pops" {
    const alloc = testing.allocator;
    var ctx = try makeTestRing(u64, alloc, 8);
    defer alloc.free(ctx.buf);

    try testing.expectEqual(@as(u32, 0), ctx.ring.len());

    // Push 3 items.
    for (0..3) |i| {
        var val: u64 = @intCast(i);
        try testing.expect(ctx.ring.tryPush(&val));
    }
    try testing.expectEqual(@as(u32, 3), ctx.ring.len());

    // Pop 1 item.
    var out: u64 = undefined;
    try testing.expect(ctx.ring.tryPop(&out));
    try testing.expectEqual(@as(u32, 2), ctx.ring.len());
}

test "extern struct items — round-trip through the ring" {
    const alloc = testing.allocator;
    var ctx = try makeTestRing(TestItem, alloc, 4);
    defer alloc.free(ctx.buf);

    var item = TestItem{ .seq_id = 42, .value = 7, .flag = 1 };
    try testing.expect(ctx.ring.tryPush(&item));

    var out: TestItem = undefined;
    try testing.expect(ctx.ring.tryPop(&out));
    try testing.expectEqual(@as(u64, 42), out.seq_id);
    try testing.expectEqual(@as(u32, 7), out.value);
    try testing.expectEqual(@as(u8, 1), out.flag);
}

test "multi-threaded stress — 100k items, no loss or corruption" {
    const alloc = testing.allocator;
    const capacity: u32 = 1024;
    const num_items: u64 = 100_000;

    var ctx = try makeTestRing(u64, alloc, capacity);
    defer alloc.free(ctx.buf);

    // Producer thread: push values 0..99999 in order.
    const producer = try std.Thread.spawn(.{}, struct {
        fn run(ring: *SpscRing(u64)) void {
            for (0..num_items) |i| {
                var val: u64 = @intCast(i);
                // Spin until there's space.
                while (!ring.tryPush(&val)) {
                    std.atomic.spinLoopHint();
                }
            }
        }
    }.run, .{&ctx.ring});

    // Consumer: pop values and verify strictly increasing order.
    var received: u64 = 0;
    while (received < num_items) {
        var out: u64 = undefined;
        if (ctx.ring.tryPop(&out)) {
            std.testing.expectEqual(received, out) catch |err| {
                std.debug.print("expected {}, got {} at index {}\n", .{ received, out, received });
                return err;
            };
            received += 1;
        } else {
            std.atomic.spinLoopHint();
        }
    }

    producer.join();

    // Ring should be empty after consuming everything.
    try testing.expectEqual(@as(u32, 0), ctx.ring.len());
}
