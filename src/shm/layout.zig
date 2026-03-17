//! Shared memory layout calculator — computes byte offsets for all
//! structures within the single mmap'd region.
//!
//! Flint uses one contiguous shared memory file per worker. This module
//! defines where each structure lives within that region, ensuring proper
//! alignment and no overlap. The layout is deterministic: given the same
//! ring capacity, both the Zig server and the Python worker compute
//! identical offsets.
//!
//! ## Memory layout
//!
//! ```
//! Offset 0                                              Total size
//! ┌───────────────────┬───────────────────┬────────────┐
//! │   Schedule Ring   │  Completion Ring  │  Heartbeat │
//! │  (Header + slots) │  (Header + slots) │  (64 bytes)│
//! └───────────────────┴───────────────────┴────────────┘
//!                     ↑                   ↑
//!              64-byte aligned      64-byte aligned
//! ```
//!
//! Each region is aligned to a 64-byte cache-line boundary to prevent
//! false sharing between the schedule ring, completion ring, and
//! heartbeat counter — each is accessed by a different producer/consumer
//! pair and may live on different CPU cores.

const std = @import("std");
const ring_buffer = @import("ring_buffer.zig");
const types = @import("types.zig");
const region_mod = @import("region.zig");
const heartbeat_mod = @import("heartbeat.zig");

const SpscRing = ring_buffer.SpscRing;
const ShmRegion = region_mod.ShmRegion;
const HeartbeatRegion = heartbeat_mod.HeartbeatRegion;

/// Cache-line size used for inter-region alignment.
const CACHE_LINE: usize = 64;

/// Round `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two. This is the standard
/// bit-manipulation trick: add (alignment - 1) then mask off the low
/// bits.
fn alignUp(value: usize, alignment: usize) usize {
    std.debug.assert(alignment > 0 and (alignment & (alignment - 1)) == 0);
    return (value + alignment - 1) & ~(alignment - 1);
}

/// Byte offsets and total size for the shared memory region.
///
/// Constructed by `compute()` and then passed to `ShmHandle.init()` to
/// overlay typed structures on the raw memory. The offsets are also
/// communicated to the Python worker (via a config struct or environment
/// variable) so it can mmap the same file and access the same regions.
pub const ShmLayout = struct {
    /// Byte offset where the schedule ring begins (always 0).
    schedule_ring_offset: usize,

    /// Byte offset where the completion ring begins.
    completion_ring_offset: usize,

    /// Byte offset where the heartbeat counter begins.
    heartbeat_offset: usize,

    /// Total number of bytes required for the entire shared memory file.
    total_size: usize,

    /// Compute the layout for the given ring capacity.
    ///
    /// All inter-region boundaries are rounded up to 64-byte alignment
    /// to prevent false sharing. The schedule ring starts at offset 0;
    /// subsequent regions follow contiguously (with alignment padding).
    pub fn compute(ring_capacity: u32) ShmLayout {
        // Schedule ring: Header (128 bytes) + capacity * @sizeOf(ScheduleT)
        const schedule_ring_size = SpscRing(types.ScheduleT).totalSize(ring_capacity);

        // Completion ring starts after the schedule ring, aligned up.
        const completion_ring_offset = alignUp(schedule_ring_size, CACHE_LINE);
        const completion_ring_size = SpscRing(types.CompletionT).totalSize(ring_capacity);

        // Heartbeat starts after the completion ring, aligned up.
        const heartbeat_offset = alignUp(completion_ring_offset + completion_ring_size, CACHE_LINE);

        // Total size includes the heartbeat region, rounded up to a
        // page boundary for clean mmap semantics.
        const raw_total = heartbeat_offset + @sizeOf(HeartbeatRegion);
        const total_size = alignUp(raw_total, std.heap.page_size_min);

        return .{
            .schedule_ring_offset = 0,
            .completion_ring_offset = completion_ring_offset,
            .heartbeat_offset = heartbeat_offset,
            .total_size = total_size,
        };
    }
};

/// Convenience handle that bundles typed views of all shared memory
/// structures.
///
/// Created by calling `init()` with a mapped `ShmRegion` and the
/// corresponding `ShmLayout`. After initialization, the caller can
/// use `schedule_ring`, `completion_ring`, and `heartbeat` directly.
pub const ShmHandle = struct {
    /// Schedule ring: Zig (producer) writes iteration schedules,
    /// Python (consumer) reads them.
    schedule_ring: SpscRing(types.ScheduleT),

    /// Completion ring: Python (producer) writes output tokens,
    /// Zig (consumer) reads them.
    completion_ring: SpscRing(types.CompletionT),

    /// Heartbeat counter: Python (writer) increments every iteration,
    /// Zig supervisor (reader) monitors for stalls.
    heartbeat: *HeartbeatRegion,

    /// Initialize typed views over a shared memory region.
    ///
    /// The caller must ensure that `region.len >= layout.total_size`.
    /// The region should be zero-initialized (e.g., freshly created
    /// via `ShmRegion.create`) so that ring buffer head/tail counters
    /// start at 0.
    pub fn init(region: *const ShmRegion, layout: ShmLayout, ring_capacity: u32) ShmHandle {
        std.debug.assert(region.len >= layout.total_size);

        return .{
            .schedule_ring = SpscRing(types.ScheduleT).init(
                region.ptr,
                layout.schedule_ring_offset,
                ring_capacity,
            ),
            .completion_ring = SpscRing(types.CompletionT).init(
                region.ptr,
                layout.completion_ring_offset,
                ring_capacity,
            ),
            .heartbeat = @ptrCast(@alignCast(region.ptr + layout.heartbeat_offset)),
        };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const linux = std.os.linux;

/// Path used by tests that need a real shared memory region.
const test_shm_path: [*:0]const u8 = "/dev/shm/flint_test_layout";

fn cleanupTestFile() void {
    _ = linux.unlink(test_shm_path);
}

test "alignUp — basic cases" {
    // Already aligned.
    try testing.expectEqual(@as(usize, 64), alignUp(64, 64));
    // Needs rounding up.
    try testing.expectEqual(@as(usize, 64), alignUp(1, 64));
    try testing.expectEqual(@as(usize, 128), alignUp(65, 64));
    // Zero stays zero.
    try testing.expectEqual(@as(usize, 0), alignUp(0, 64));
    // Power-of-two alignments other than 64.
    try testing.expectEqual(@as(usize, 4096), alignUp(3000, 4096));
}

test "layout — all offsets are 64-byte aligned" {
    const layout = ShmLayout.compute(64);

    try testing.expectEqual(@as(usize, 0), layout.schedule_ring_offset % CACHE_LINE);
    try testing.expectEqual(@as(usize, 0), layout.completion_ring_offset % CACHE_LINE);
    try testing.expectEqual(@as(usize, 0), layout.heartbeat_offset % CACHE_LINE);
    // Total size is page-aligned (which is a multiple of 64).
    try testing.expectEqual(@as(usize, 0), layout.total_size % std.heap.page_size_min);
}

test "layout — regions do not overlap" {
    const capacity: u32 = 64;
    const layout = ShmLayout.compute(capacity);

    const schedule_ring_end = layout.schedule_ring_offset +
        SpscRing(types.ScheduleT).totalSize(capacity);
    const completion_ring_end = layout.completion_ring_offset +
        SpscRing(types.CompletionT).totalSize(capacity);
    const heartbeat_end = layout.heartbeat_offset + @sizeOf(HeartbeatRegion);

    // Schedule ring ends before completion ring starts.
    try testing.expect(schedule_ring_end <= layout.completion_ring_offset);

    // Completion ring ends before heartbeat starts.
    try testing.expect(completion_ring_end <= layout.heartbeat_offset);

    // Heartbeat ends within the total allocation.
    try testing.expect(heartbeat_end <= layout.total_size);
}

test "layout — total_size is large enough for all regions" {
    // Try several capacities.
    for ([_]u32{ 1, 4, 16, 64, 256 }) |cap| {
        const layout = ShmLayout.compute(cap);

        const needed = layout.heartbeat_offset + @sizeOf(HeartbeatRegion);
        try testing.expect(layout.total_size >= needed);
    }
}

test "full init roundtrip — push and pop a CompletionT through shm" {
    cleanupTestFile();
    defer cleanupTestFile();

    const capacity: u32 = 16;
    const layout = ShmLayout.compute(capacity);

    // Create a real shared memory region.
    var shm = try ShmRegion.create(test_shm_path, layout.total_size);
    defer shm.close();

    // Zero-initialize the region (ring head/tail must start at 0).
    @memset(shm.ptr[0..shm.len], 0);

    var handle = ShmHandle.init(&shm, layout, capacity);

    // Push a completion.
    var completion = types.CompletionT{
        .seq_id = 42,
        .token_id = 1337,
        .logprob = @as(f16, -1.5),
        .is_eos = 1,
    };
    try testing.expect(handle.completion_ring.tryPush(&completion));

    // Pop it back and verify every field.
    var out: types.CompletionT = undefined;
    try testing.expect(handle.completion_ring.tryPop(&out));
    try testing.expectEqual(@as(u64, 42), out.seq_id);
    try testing.expectEqual(@as(u32, 1337), out.token_id);
    try testing.expectEqual(@as(f16, -1.5), out.logprob);
    try testing.expectEqual(@as(u8, 1), out.is_eos);
}

test "full init roundtrip — heartbeat through shm" {
    cleanupTestFile();
    defer cleanupTestFile();

    const capacity: u32 = 16;
    const layout = ShmLayout.compute(capacity);

    var shm = try ShmRegion.create(test_shm_path, layout.total_size);
    defer shm.close();
    @memset(shm.ptr[0..shm.len], 0);

    const handle = ShmHandle.init(&shm, layout, capacity);

    // Heartbeat starts at zero.
    try testing.expectEqual(@as(u64, 0), handle.heartbeat.read());

    // Simulate worker heartbeats.
    handle.heartbeat.increment();
    handle.heartbeat.increment();
    handle.heartbeat.increment();
    try testing.expectEqual(@as(u64, 3), handle.heartbeat.read());

    // Reset (e.g., on worker restart).
    handle.heartbeat.reset();
    try testing.expectEqual(@as(u64, 0), handle.heartbeat.read());
}

test "full init roundtrip — schedule ring push and pop" {
    cleanupTestFile();
    defer cleanupTestFile();

    const capacity: u32 = 4;
    const layout = ShmLayout.compute(capacity);

    var shm = try ShmRegion.create(test_shm_path, layout.total_size);
    defer shm.close();
    @memset(shm.ptr[0..shm.len], 0);

    var handle = ShmHandle.init(&shm, layout, capacity);

    // Push a minimal schedule.
    var sched = std.mem.zeroes(types.ScheduleT);
    sched.iteration_id = 7;
    sched.num_sequences = 1;
    sched.seq_ids[0] = 99;
    sched.token_ids[0] = 555;

    try testing.expect(handle.schedule_ring.tryPush(&sched));

    // Pop and verify.
    var out: types.ScheduleT = undefined;
    try testing.expect(handle.schedule_ring.tryPop(&out));
    try testing.expectEqual(@as(u64, 7), out.iteration_id);
    try testing.expectEqual(@as(u32, 1), out.num_sequences);
    try testing.expectEqual(@as(u64, 99), out.seq_ids[0]);
    try testing.expectEqual(@as(u32, 555), out.token_ids[0]);
}
