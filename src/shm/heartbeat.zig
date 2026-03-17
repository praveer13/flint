//! Worker liveness heartbeat — an atomic counter in shared memory.
//!
//! The heartbeat protocol is simple:
//!
//!   1. The Zig supervisor spawns a Python GPU worker and points it at a
//!      shared memory region containing a `HeartbeatRegion`.
//!   2. The worker calls `increment()` after every forward-pass iteration.
//!   3. The supervisor periodically calls `read()` and compares the value
//!      to the previous sample. If the counter has not advanced after N
//!      consecutive checks, the worker is presumed dead (hung in a CUDA
//!      call, segfaulted, etc.) and the supervisor initiates recovery.
//!
//! Why not a timestamp? Because `CLOCK_MONOTONIC` across processes on
//! shared memory requires careful handling of clock resolution and is
//! overkill for a liveness check. A simple counter that must advance is
//! sufficient — the supervisor controls the polling interval, so it
//! implicitly defines the timeout.
//!
//! Memory layout: a single 64-byte cache-line-aligned region containing
//! one `u64` counter plus padding. The padding prevents false sharing if
//! adjacent shared memory structures are accessed by different cores.

const std = @import("std");

/// Atomic heartbeat counter residing in shared memory.
///
/// Layout (64 bytes total, cache-line aligned):
/// ```
/// offset 0:   counter (u64)   — atomically incremented by the worker
/// offset 8:   _pad (56 bytes) — fills the rest of the cache line
/// ```
///
/// This is an `extern struct` so its layout is deterministic and
/// reproducible in Python (via `struct.pack_into` / `struct.unpack_from`
/// at the known offset).
pub const HeartbeatRegion = extern struct {
    /// Monotonically increasing counter. The worker increments this after
    /// every iteration; the supervisor reads it to detect stalls.
    counter: u64 align(64),

    /// Padding to fill the 64-byte cache line. Prevents false sharing
    /// with adjacent structures in the shared memory region.
    _pad: [56]u8 = .{0} ** 56,

    /// Bump the heartbeat counter by one.
    ///
    /// Called by the worker after each forward-pass iteration. Uses
    /// monotonic ordering because correctness does not depend on the
    /// supervisor seeing this increment in any particular order relative
    /// to other memory writes — the supervisor only cares that the value
    /// *eventually* advances.
    pub fn increment(self: *HeartbeatRegion) void {
        _ = @atomicRmw(u64, &self.counter, .Add, 1, .monotonic);
    }

    /// Read the current heartbeat value.
    ///
    /// Called by the supervisor to sample the counter. The supervisor
    /// compares consecutive samples: if `read()` returns the same value
    /// across multiple polling intervals, the worker is presumed dead.
    ///
    /// Uses monotonic ordering — we do not need to synchronize with any
    /// other memory locations, just observe that the counter moves.
    pub fn read(self: *const HeartbeatRegion) u64 {
        return @atomicLoad(u64, &self.counter, .monotonic);
    }

    /// Reset the counter to zero.
    ///
    /// Called by the Zig server when initializing or re-initializing a
    /// worker's shared memory region (e.g., after a worker restart).
    pub fn reset(self: *HeartbeatRegion) void {
        @atomicStore(u64, &self.counter, 0, .monotonic);
    }
};

// Compile-time layout assertions.
comptime {
    const assert = std.debug.assert;

    // Must occupy exactly one cache line (64 bytes).
    assert(@sizeOf(HeartbeatRegion) == 64);

    // Counter must be at offset 0.
    assert(@offsetOf(HeartbeatRegion, "counter") == 0);

    // Must be 64-byte aligned (cache-line boundary).
    assert(@alignOf(HeartbeatRegion) == 64);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "increment advances the counter" {
    var hb: HeartbeatRegion = .{ .counter = 0 };

    try testing.expectEqual(@as(u64, 0), hb.read());

    hb.increment();
    try testing.expectEqual(@as(u64, 1), hb.read());

    hb.increment();
    hb.increment();
    try testing.expectEqual(@as(u64, 3), hb.read());
}

test "reset sets counter to zero" {
    var hb: HeartbeatRegion = .{ .counter = 0 };

    hb.increment();
    hb.increment();
    try testing.expectEqual(@as(u64, 2), hb.read());

    hb.reset();
    try testing.expectEqual(@as(u64, 0), hb.read());
}

test "initial zero-initialized state reads as zero" {
    const hb = std.mem.zeroes(HeartbeatRegion);
    try testing.expectEqual(@as(u64, 0), hb.read());
}

test "many increments do not overflow in practice" {
    var hb: HeartbeatRegion = .{ .counter = 0 };

    // Simulate 1000 iterations.
    for (0..1000) |_| {
        hb.increment();
    }
    try testing.expectEqual(@as(u64, 1000), hb.read());
}
