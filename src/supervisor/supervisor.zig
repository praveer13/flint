//! Worker process lifecycle supervisor.
//!
//! Flint delegates GPU computation to separate Python worker processes that
//! import vLLM as a library. Each worker is pinned to a specific GPU and
//! communicates with the Zig server exclusively through shared memory —
//! no sockets, no RPC, no serialization. The supervisor is the Zig-side
//! component responsible for spawning these workers, monitoring their
//! liveness, and detecting when they die or stall.
//!
//! ## Why separate processes?
//!
//! 1. **Python GIL isolation.** Each worker is its own Python interpreter.
//!    There is no GIL contention because there is exactly one thread per
//!    worker process. Shared memory bypasses the GIL entirely — both sides
//!    read and write through `mmap`'d pointers.
//!
//! 2. **GPU fault isolation.** A CUDA error (OOM, ECC failure, driver crash)
//!    kills only the affected worker process. The Zig server survives, can
//!    preempt the affected sequences, and respawn the worker on the same GPU
//!    without restarting the entire service.
//!
//! 3. **Clean lifecycle management.** Workers are stateless (all persistent
//!    state is in Zig-owned shared memory). Killing and restarting a worker
//!    is safe — the new worker re-attaches to the same shm region and
//!    resumes from the current schedule ring position.
//!
//! ## Lifecycle state machine
//!
//! ```
//! not_started ─► starting ─► running ─► dead
//!                                │          ▲
//!                                └──────────┘
//!                              (crash / stall)
//! ```
//!
//!   - `not_started`: Initial state. No process has been spawned for this GPU.
//!   - `starting`: The worker process has been spawned but has not yet
//!     signalled readiness via the heartbeat counter. The supervisor waits
//!     for the heartbeat to advance from zero, indicating the worker has
//!     finished initialization (model load, CUDA context setup, etc.).
//!   - `running`: The heartbeat is advancing normally. The worker is
//!     processing schedules and producing completions.
//!   - `dead`: The worker process has exited (detected via `waitpid` with
//!     `WNOHANG`) or its heartbeat has not advanced within the stall
//!     timeout (default 5 seconds). The supervisor marks the worker dead;
//!     the caller can decide whether to respawn.
//!
//! ## Health monitoring
//!
//! The supervisor uses shared-memory heartbeat counters (see `shm/heartbeat.zig`)
//! rather than TCP health checks. The heartbeat is an atomic `u64` in shared
//! memory that the worker increments after every forward-pass iteration. The
//! supervisor samples this counter periodically:
//!
//!   - If the counter has advanced since the last check → worker is alive.
//!   - If the counter has not advanced for longer than `stall_timeout_ns` →
//!     worker is presumed dead (likely stuck in a CUDA call or segfaulted).
//!
//! This avoids the overhead and latency of TCP round-trips and works even
//! when the worker's Python process is hung in native code (where it would
//! not respond to a TCP health check anyway).
//!
//! ## Phase 4 simplifications
//!
//! - Single worker (gpu_id=0) only.
//! - No automatic respawn — detection only. The caller can inspect state
//!   via `getWorkerState` and decide whether to call `spawnWorker` again.
//! - Stall timeout is a fixed 5 seconds.

const std = @import("std");
const HeartbeatRegion = @import("../shm/heartbeat.zig").HeartbeatRegion;

const linux = std.os.linux;

/// Default stall timeout: if a worker's heartbeat does not advance for this
/// duration, the worker is considered dead.
const DEFAULT_STALL_TIMEOUT_NS: i128 = 5 * std.time.ns_per_s;

/// Worker process lifecycle supervisor.
///
/// Manages one `Worker` entry per GPU. Call `spawnWorker` to launch a
/// Python worker process for a given GPU, then call `checkHealth`
/// periodically (e.g., every 500 ms) to detect crashes and stalls.
pub const WorkerSupervisor = struct {
    /// Per-GPU worker state.
    workers: []Worker,

    /// Filesystem path to the shared memory file that both the Zig server
    /// and the Python workers mmap. Passed to the worker as a CLI argument.
    shm_path: []const u8,

    /// Path to the Python worker script (e.g., `python/flint_worker.py`).
    worker_script: []const u8,

    /// Pointer to the heartbeat region in shared memory. The supervisor
    /// reads from this; workers write to it.
    heartbeat: *HeartbeatRegion,

    /// Stall timeout in nanoseconds. If a worker's heartbeat has not
    /// advanced within this duration, it is marked dead.
    stall_timeout_ns: i128,

    /// Lifecycle state for a single GPU worker process.
    pub const WorkerState = enum {
        /// No process has been spawned yet for this GPU.
        not_started,
        /// Process has been spawned; waiting for heartbeat to indicate ready.
        starting,
        /// Heartbeat is advancing normally — worker is healthy.
        running,
        /// Process exited or heartbeat stalled — worker is dead.
        dead,
    };

    /// Tracks the state of one worker process.
    pub const Worker = struct {
        /// Handle to the child process, or null if not spawned / already waited.
        child: ?std.process.Child,
        /// Which GPU this worker is pinned to.
        gpu_id: u32,
        /// Current lifecycle state.
        state: WorkerState,
        /// Last observed heartbeat counter value.
        last_heartbeat: u64,
        /// Timestamp (from `CLOCK_MONOTONIC`, in nanoseconds) of the last
        /// health check where the heartbeat was observed to have advanced.
        /// Used for stall detection.
        last_check_time_ns: i128,
    };

    /// Initialize a supervisor over the provided worker buffer.
    ///
    /// `workers_buf` must have one entry per GPU. The supervisor does not
    /// allocate — it borrows this slice for the lifetime of the supervisor.
    pub fn init(
        workers_buf: []Worker,
        shm_path: []const u8,
        worker_script: []const u8,
        heartbeat: *HeartbeatRegion,
    ) WorkerSupervisor {
        return initWithTimeout(workers_buf, shm_path, worker_script, heartbeat, DEFAULT_STALL_TIMEOUT_NS);
    }

    /// Initialize with a custom stall timeout (useful for tests).
    pub fn initWithTimeout(
        workers_buf: []Worker,
        shm_path: []const u8,
        worker_script: []const u8,
        heartbeat: *HeartbeatRegion,
        stall_timeout_ns: i128,
    ) WorkerSupervisor {
        // Zero-initialize all worker entries.
        for (workers_buf) |*w| {
            w.* = .{
                .child = null,
                .gpu_id = 0,
                .state = .not_started,
                .last_heartbeat = 0,
                .last_check_time_ns = 0,
            };
        }

        return .{
            .workers = workers_buf,
            .shm_path = shm_path,
            .worker_script = worker_script,
            .heartbeat = heartbeat,
            .stall_timeout_ns = stall_timeout_ns,
        };
    }

    /// Spawn a worker process for the given GPU.
    ///
    /// The worker is launched as:
    /// ```
    /// python3 <worker_script> <shm_path> --gpu <gpu_id>
    /// ```
    ///
    /// The heartbeat region is reset before spawning so that the supervisor
    /// can distinguish a fresh worker's first heartbeat from stale data.
    ///
    /// Requires `io` because `std.process.spawn` dispatches through the
    /// `std.Io` vtable (on `Io.Evented` it may use io_uring for `clone`;
    /// on `Io.Threaded` it uses the POSIX `fork`/`exec` path).
    pub fn spawnWorker(self: *WorkerSupervisor, io: std.Io, gpu_id: u32) !void {
        if (gpu_id >= self.workers.len) return error.InvalidGpuId;

        const w = &self.workers[gpu_id];

        // If a previous child is still lingering, clean it up.
        if (w.child != null) {
            w.child.?.kill(io);
            w.child = null;
        }

        // Reset heartbeat so we can detect when the new worker signals ready.
        self.heartbeat.reset();

        // Format gpu_id as a string for the argv. We use a small stack
        // buffer — gpu_id is a u32, so at most 10 decimal digits.
        var gpu_id_buf: [16]u8 = undefined;
        const gpu_id_str = std.fmt.bufPrint(&gpu_id_buf, "{d}", .{gpu_id}) catch unreachable;

        const argv: []const []const u8 = &.{
            "python3",
            self.worker_script,
            self.shm_path,
            "--gpu",
            gpu_id_str,
        };

        var child = try std.process.spawn(io, .{
            .argv = argv,
            .stdin = .inherit,
            .stdout = .inherit,
            .stderr = .inherit,
        });

        // Record as started; we don't own stdin/stdout/stderr pipes.
        _ = child.stdin;
        _ = child.stdout;
        _ = child.stderr;

        w.child = child;
        w.gpu_id = gpu_id;
        w.state = .starting;
        w.last_heartbeat = 0;
        w.last_check_time_ns = monotonicNowNs();
    }

    /// Check the health of all workers. Call this periodically.
    ///
    /// For each worker that is `starting` or `running`:
    ///   1. Check if the OS process has exited (via `waitpid` with `WNOHANG`).
    ///      If so, mark as `dead`.
    ///   2. Read the heartbeat counter from shared memory.
    ///      - If the counter has advanced, update `last_heartbeat` and
    ///        `last_check_time_ns`, and transition `starting` → `running`.
    ///      - If the counter has not advanced and the stall timeout has
    ///        elapsed, mark as `dead`.
    ///
    /// Returns `true` if all spawned workers are healthy (running), `false`
    /// if any worker is dead.
    pub fn checkHealth(self: *WorkerSupervisor) bool {
        var all_healthy = true;
        const now_ns = monotonicNowNs();

        for (self.workers) |*w| {
            switch (w.state) {
                .not_started => continue,
                .dead => {
                    all_healthy = false;
                    continue;
                },
                .starting, .running => {},
            }

            // 1. Check if the process has exited.
            if (w.child) |child| {
                if (child.id) |pid| {
                    if (processHasExited(pid)) {
                        w.state = .dead;
                        // Don't call wait() here (it blocks through Io).
                        // The child handle is kept so shutdown() can reap it.
                        all_healthy = false;
                        continue;
                    }
                }
            }

            // 2. Check heartbeat progress.
            const current_hb = self.heartbeat.read();

            if (current_hb != w.last_heartbeat) {
                // Heartbeat advanced — worker is alive.
                w.last_heartbeat = current_hb;
                w.last_check_time_ns = now_ns;

                // Transition starting → running on first heartbeat.
                if (w.state == .starting) {
                    w.state = .running;
                }
            } else {
                // Heartbeat stalled — check timeout.
                const elapsed = now_ns - w.last_check_time_ns;
                if (elapsed >= self.stall_timeout_ns) {
                    w.state = .dead;
                    all_healthy = false;
                }
            }
        }

        return all_healthy;
    }

    /// Send a termination signal to all workers and wait for them to exit.
    ///
    /// This is a best-effort cleanup. Workers that do not exit within a
    /// reasonable time are killed with `SIGKILL` (via `child.kill()`).
    ///
    /// After `shutdown`, all workers are in the `dead` state and their
    /// child handles are nulled out.
    pub fn shutdown(self: *WorkerSupervisor, io: std.Io) void {
        for (self.workers) |*w| {
            if (w.child) |*child| {
                // kill() sends SIGKILL and blocks until the child exits,
                // then sets child.id to null.
                child.kill(io);
                w.child = null;
            }
            if (w.state != .not_started) {
                w.state = .dead;
            }
        }
    }

    /// Get the current lifecycle state of a worker.
    pub fn getWorkerState(self: *const WorkerSupervisor, gpu_id: u32) WorkerState {
        if (gpu_id >= self.workers.len) return .not_started;
        return self.workers[gpu_id].state;
    }

    /// Check whether a process has exited without blocking, using raw
    /// `waitpid` with `WNOHANG`. Returns `true` if the process has exited
    /// (or the pid is invalid), `false` if it is still running.
    fn processHasExited(pid: linux.pid_t) bool {
        var status: u32 = 0;
        const rc = linux.waitpid(pid, &status, linux.W.NOHANG);
        const err = linux.errno(rc);

        if (err != .SUCCESS) {
            // ECHILD means the child doesn't exist (already reaped, or not ours).
            // Treat any error as "process gone".
            return true;
        }

        // rc == 0 means child is still running (WNOHANG returned immediately).
        // rc > 0 means child exited / was signalled.
        return rc != 0;
    }

    /// Read `CLOCK_MONOTONIC` in nanoseconds.
    fn monotonicNowNs() i128 {
        var ts: linux.timespec = undefined;
        // clock_gettime with CLOCK_MONOTONIC always succeeds on Linux.
        _ = linux.clock_gettime(.MONOTONIC, &ts);
        return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "init — all workers start in not_started state" {
    var hb: HeartbeatRegion = .{ .counter = 0 };
    var workers_buf: [4]WorkerSupervisor.Worker = undefined;

    const sup = WorkerSupervisor.init(
        &workers_buf,
        "/dev/shm/test_flint",
        "python/mock_worker.py",
        &hb,
    );

    for (sup.workers) |w| {
        try testing.expectEqual(WorkerSupervisor.WorkerState.not_started, w.state);
        try testing.expect(w.child == null);
    }
}

test "getWorkerState — returns not_started for unspawned workers" {
    var hb: HeartbeatRegion = .{ .counter = 0 };
    var workers_buf: [2]WorkerSupervisor.Worker = undefined;

    const sup = WorkerSupervisor.init(
        &workers_buf,
        "/dev/shm/test_flint",
        "python/mock_worker.py",
        &hb,
    );

    try testing.expectEqual(WorkerSupervisor.WorkerState.not_started, sup.getWorkerState(0));
    try testing.expectEqual(WorkerSupervisor.WorkerState.not_started, sup.getWorkerState(1));
    // Out-of-bounds gpu_id returns not_started gracefully.
    try testing.expectEqual(WorkerSupervisor.WorkerState.not_started, sup.getWorkerState(99));
}

test "checkHealth — no-op when no workers spawned" {
    var hb: HeartbeatRegion = .{ .counter = 0 };
    var workers_buf: [1]WorkerSupervisor.Worker = undefined;

    var sup = WorkerSupervisor.init(
        &workers_buf,
        "/dev/shm/test_flint",
        "python/mock_worker.py",
        &hb,
    );

    // With no workers spawned, checkHealth should report all healthy (vacuously true).
    try testing.expect(sup.checkHealth());
}

test "checkHealth — detects heartbeat advancement (starting → running)" {
    var hb: HeartbeatRegion = .{ .counter = 0 };
    var workers_buf: [1]WorkerSupervisor.Worker = undefined;

    var sup = WorkerSupervisor.init(
        &workers_buf,
        "/dev/shm/test_flint",
        "python/mock_worker.py",
        &hb,
    );

    // Simulate a worker that has been spawned (manually set state to starting,
    // without actually forking — we test the state machine logic in isolation).
    workers_buf[0].state = .starting;
    workers_buf[0].last_heartbeat = 0;
    workers_buf[0].last_check_time_ns = WorkerSupervisor.monotonicNowNs();
    // No real child process — set child to null so processHasExited is skipped.
    workers_buf[0].child = null;

    // Before heartbeat advances, state stays starting.
    try testing.expectEqual(WorkerSupervisor.WorkerState.starting, sup.getWorkerState(0));

    // Simulate the worker writing its first heartbeat.
    hb.increment();

    _ = sup.checkHealth();
    try testing.expectEqual(WorkerSupervisor.WorkerState.running, sup.getWorkerState(0));
}

test "checkHealth — detects stall (running → dead)" {
    var hb: HeartbeatRegion = .{ .counter = 0 };
    var workers_buf: [1]WorkerSupervisor.Worker = undefined;

    // Use a very short stall timeout for the test (1 ns).
    var sup = WorkerSupervisor.initWithTimeout(
        &workers_buf,
        "/dev/shm/test_flint",
        "python/mock_worker.py",
        &hb,
        1, // 1 ns timeout — will trigger immediately on next check.
    );

    // Simulate a running worker whose heartbeat is stale.
    workers_buf[0].state = .running;
    workers_buf[0].last_heartbeat = 5;
    workers_buf[0].last_check_time_ns = 0; // Long ago.
    workers_buf[0].child = null;
    hb.counter = 5; // Same as last_heartbeat — stalled.

    const healthy = sup.checkHealth();
    try testing.expect(!healthy);
    try testing.expectEqual(WorkerSupervisor.WorkerState.dead, sup.getWorkerState(0));
}

test "checkHealth — healthy worker keeps running" {
    var hb: HeartbeatRegion = .{ .counter = 0 };
    var workers_buf: [1]WorkerSupervisor.Worker = undefined;

    var sup = WorkerSupervisor.init(
        &workers_buf,
        "/dev/shm/test_flint",
        "python/mock_worker.py",
        &hb,
    );

    // Simulate a running worker with a fresh heartbeat.
    workers_buf[0].state = .running;
    workers_buf[0].last_heartbeat = 10;
    workers_buf[0].last_check_time_ns = WorkerSupervisor.monotonicNowNs();
    workers_buf[0].child = null;

    // Advance heartbeat — worker is alive.
    hb.counter = 11;

    const healthy = sup.checkHealth();
    try testing.expect(healthy);
    try testing.expectEqual(WorkerSupervisor.WorkerState.running, sup.getWorkerState(0));
    try testing.expectEqual(@as(u64, 11), workers_buf[0].last_heartbeat);
}

test "processHasExited — returns true for invalid pid" {
    // PID -1 is invalid; waitpid should return ECHILD → treated as exited.
    try testing.expect(WorkerSupervisor.processHasExited(-1));
}

test "monotonicNowNs — returns a positive timestamp" {
    const now = WorkerSupervisor.monotonicNowNs();
    try testing.expect(now > 0);
}
