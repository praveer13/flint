//! Central scheduling engine for Flint's inference pipeline.
//!
//! The scheduler is the bridge between the HTTP layer (which receives inference
//! requests) and the GPU worker (which executes forward passes). It runs on a
//! **dedicated OS thread** — not an `std.Io` fiber — because its core loop
//! spin-waits on shared-memory atomics, which would block the event loop if
//! run as a fiber.
//!
//! ## Iteration loop
//!
//! The scheduler runs a tight loop, executing one "iteration" per GPU forward
//! pass cycle (~10-50 ms). Each iteration:
//!
//!   1. **Drain completions** — read output tokens from the completion ring
//!      (written by the GPU worker). Update sequence state: increment
//!      `generated_tokens`, check for EOS / max_tokens, free blocks for
//!      finished sequences.
//!
//!   2. **Drain new requests** — pull pending requests from the request queue
//!      (submitted by HTTP fibers via `submitRequest`). Add each as a new
//!      sequence in the `waiting` state.
//!
//!   3. **Compute schedule** — decide which sequences to include in this
//!      iteration's GPU batch:
//!        a. Continue all currently running sequences (allocate a new block
//!           if the sequence has filled its current one).
//!        b. Promote waiting sequences to running if free blocks are available.
//!        c. Build a `ScheduleT` struct with all running sequences' metadata.
//!      Phase 3 uses FCFS (first-come-first-served) scheduling with no
//!      preemption — if blocks run out, waiting sequences simply stay queued.
//!
//!   4. **Post schedule** — fence the block table, then push the `ScheduleT`
//!      onto the schedule ring for the GPU worker to consume.
//!
//! ## Thread safety
//!
//! The scheduler's internal state (sequence table, block allocator, block
//! table, iteration counter) is accessed only from the scheduler thread —
//! no locking needed for those.
//!
//! Two boundaries require synchronization:
//!
//!   - **Request submission** (`pending_requests`): protected by an
//!     `std.atomic.Mutex`. HTTP fibers call `submitRequest()` which acquires
//!     the mutex, appends to a bounded array, and releases. The scheduler
//!     drains this array under the same mutex once per iteration. This is not
//!     on the hot path — requests arrive at ~1000/s, not millions/s.
//!
//!   - **Completion notification** (`seq_status`): an array of per-sequence
//!     status structs with atomically-updated fields. The scheduler writes
//!     (from its thread) and HTTP fibers read (from Io fibers) using atomic
//!     load/store. No mutex needed — single writer, multiple readers.
//!
//! ## Phase 3 simplifications
//!
//! - FCFS scheduling, no priority ordering.
//! - No preemption — if out of blocks, waiting sequences stay queued.
//! - No swap commands (swap_out/swap_in left empty in ScheduleT).
//! - Fixed block allocation: `ceil(prompt_tokens / TOKENS_PER_BLOCK)` on
//!   first schedule, then one new block per iteration as tokens fill up.
//! - Single GPU worker (one schedule ring, one completion ring).

const std = @import("std");
const sequence_mod = @import("sequence.zig");
const allocator_mod = @import("../block_mgr/allocator.zig");
const block_table_mod = @import("../block_mgr/block_table.zig");
const ring_buffer_mod = @import("../shm/ring_buffer.zig");
const types = @import("../shm/types.zig");

const Sequence = sequence_mod.Sequence;
const SequenceState = sequence_mod.SequenceState;
const SequenceTable = sequence_mod.SequenceTable;
const BlockAllocator = allocator_mod.BlockAllocator;
const BlockTable = block_table_mod.BlockTable;
const SpscRing = ring_buffer_mod.SpscRing;
const ScheduleT = types.ScheduleT;
const CompletionT = types.CompletionT;
const MAX_BATCH = types.MAX_BATCH;
const MAX_BLOCKS_PER_SEQ = types.MAX_BLOCKS_PER_SEQ;

/// Number of tokens per KV-cache block. Matches the vLLM default.
const TOKENS_PER_BLOCK: u32 = 16;

/// Maximum number of pending requests that can be buffered between HTTP
/// fibers and the scheduler thread.
pub const MAX_PENDING_REQUESTS: u32 = 256;

/// Maximum number of sequences the scheduler can track simultaneously.
/// This also bounds the number of rows in the block table and the size
/// of the `seq_status` array.
pub const MAX_SEQUENCES: u32 = 1024;

// ---------------------------------------------------------------------------
// PendingRequest
// ---------------------------------------------------------------------------

/// A request submitted by an HTTP fiber, waiting to be picked up by the
/// scheduler thread.
pub const PendingRequest = struct {
    num_prompt_tokens: u32,
    max_tokens: u32,
    temperature: f16,
    top_p: f16,
    priority: u8,
};

// ---------------------------------------------------------------------------
// SequenceStatus
// ---------------------------------------------------------------------------

/// Per-sequence status visible to HTTP fibers. The scheduler writes these
/// atomically from its dedicated thread; HTTP fibers poll-read them to
/// detect new tokens and sequence completion.
///
/// Indexed by *seq_slot* (the sequence's position in the SequenceTable),
/// not by seq_id. The caller receives the slot index from `submitRequest`.
pub const SequenceStatus = struct {
    /// The seq_id assigned by the scheduler. 0 means the slot is unused.
    /// Written once by the scheduler when the request is admitted.
    seq_id: u64 = 0,

    /// The most recently generated token ID. Updated by the scheduler each
    /// time it drains a completion for this sequence.
    last_token_id: u32 = 0,

    /// Total tokens generated so far. Monotonically increasing.
    tokens_generated: u32 = 0,

    /// 1 when the sequence has finished (EOS or max_tokens). Once set to 1,
    /// never reverts to 0 for this slot (until the slot is recycled).
    is_done: u8 = 0,
};

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// The central scheduling engine.
///
/// Create with `init`, start the background thread with `start`, submit
/// requests with `submitRequest`, poll completion status with `getStatus`,
/// and stop with `stop`.
pub const Scheduler = struct {
    // -- Core state (scheduler-thread only, no locking) --

    /// Tracks all active sequences.
    sequences: SequenceTable,

    /// Manages free KV-cache block IDs.
    block_alloc: BlockAllocator,

    /// Maps (seq_slot, logical_block) -> physical block ID in shared memory.
    block_table: BlockTable,

    /// SPSC ring: scheduler (producer) -> GPU worker (consumer).
    schedule_ring: SpscRing(ScheduleT),

    /// SPSC ring: GPU worker (producer) -> scheduler (consumer).
    completion_ring: SpscRing(CompletionT),

    /// Monotonically increasing iteration counter.
    iteration: u64,

    // -- Request submission (mutex-protected) --

    /// Guards `pending_requests` and `pending_count`.
    request_mutex: std.atomic.Mutex,

    /// Bounded buffer of requests waiting to be drained by the scheduler.
    pending_requests: [MAX_PENDING_REQUESTS]PendingRequest,

    /// Number of valid entries in `pending_requests[0..pending_count]`.
    pending_count: u32,

    // -- Completion notification (atomic, polled by HTTP fibers) --

    /// Per-slot status. Indexed by the slot index returned from `submitRequest`.
    seq_status: []SequenceStatus,

    // -- Thread control --

    /// Handle to the scheduler's OS thread, set by `start`.
    thread: ?std.Thread,

    /// Set to true (release) to request the scheduler loop to exit.
    /// The scheduler checks this (acquire) at the top of each iteration.
    should_stop: bool,

    // -- Mapping from seq_id -> seq_slot --
    //
    // The SequenceTable uses swap-removal, so the *array index* of a
    // sequence can change. We need a stable slot index for the block table
    // and seq_status arrays. We maintain a simple map: seq_slot is assigned
    // at admission time and stays constant until the sequence finishes.
    // seq_id_to_slot[i] holds the seq_id for slot i (0 = unused).
    seq_id_to_slot: []u64,

    /// Next slot index to try when assigning a new sequence slot.
    next_slot_hint: u32,

    /// Maximum number of sequence slots available.
    max_seq_slots: u32,

    /// Parallel array for slot assignments corresponding to pending_requests.
    /// Protected by `request_mutex` alongside `pending_requests` and
    /// `pending_count`.
    pending_slots: [MAX_PENDING_REQUESTS]u32,

    /// Initialize the scheduler.
    ///
    /// All backing memory is provided by the caller — the scheduler does
    /// not allocate. This makes it embeddable in shared memory or any
    /// fixed-size arena.
    ///
    /// Parameters:
    /// - `seq_backing`: Backing array for the SequenceTable.
    /// - `free_stack_buf` / `ref_count_buf`: Backing for the BlockAllocator.
    /// - `total_blocks`: Number of KV-cache blocks to manage.
    /// - `block_table_base` / `block_table_offset`: Shared memory location for
    ///    the block table.
    /// - `max_blocks_per_seq`: Max logical blocks per sequence (columns in
    ///    block table).
    /// - `schedule_ring_base` / `schedule_ring_offset` / `schedule_ring_cap`:
    ///    Shared memory for the schedule ring.
    /// - `completion_ring_base` / `completion_ring_offset` / `completion_ring_cap`:
    ///    Shared memory for the completion ring.
    /// - `seq_status_buf`: Backing for per-sequence status (length >= seq_backing.len).
    /// - `slot_map_buf`: Backing for seq_id-to-slot mapping (length >= seq_backing.len).
    pub fn init(
        seq_backing: []Sequence,
        free_stack_buf: []u32,
        ref_count_buf: []u16,
        total_blocks: u32,
        block_table_base: [*]u8,
        block_table_offset: usize,
        max_blocks_per_seq: u32,
        schedule_ring_base: [*]u8,
        schedule_ring_offset: usize,
        schedule_ring_cap: u32,
        completion_ring_base: [*]u8,
        completion_ring_offset: usize,
        completion_ring_cap: u32,
        seq_status_buf: []SequenceStatus,
        slot_map_buf: []u64,
    ) Scheduler {
        const max_seqs: u32 = @intCast(seq_backing.len);

        // Zero out the slot map.
        @memset(slot_map_buf[0..max_seqs], 0);

        // Zero out the status buffer.
        for (seq_status_buf[0..max_seqs]) |*s| {
            s.* = .{};
        }

        return .{
            .sequences = SequenceTable.init(seq_backing),
            .block_alloc = BlockAllocator.init(total_blocks, free_stack_buf, ref_count_buf),
            .block_table = BlockTable.init(block_table_base, block_table_offset, max_seqs, max_blocks_per_seq),
            .schedule_ring = SpscRing(ScheduleT).init(schedule_ring_base, schedule_ring_offset, schedule_ring_cap),
            .completion_ring = SpscRing(CompletionT).init(completion_ring_base, completion_ring_offset, completion_ring_cap),
            .iteration = 0,
            .request_mutex = .unlocked,
            .pending_requests = undefined,
            .pending_slots = undefined,
            .pending_count = 0,
            .seq_status = seq_status_buf[0..max_seqs],
            .thread = null,
            .should_stop = false,
            .seq_id_to_slot = slot_map_buf[0..max_seqs],
            .next_slot_hint = 0,
            .max_seq_slots = max_seqs,
        };
    }

    /// Start the scheduler on a dedicated OS thread.
    ///
    /// The thread runs `schedulerLoop` until `stop()` is called.
    pub fn start(self: *Scheduler) std.Thread.SpawnError!void {
        self.thread = try std.Thread.spawn(.{}, schedulerLoop, .{self});
    }

    /// Stop the scheduler thread gracefully.
    ///
    /// Signals the loop to exit, then joins the thread. After this call,
    /// the scheduler is inert — `submitRequest` still works but nothing
    /// will drain the pending queue.
    pub fn stop(self: *Scheduler) void {
        @atomicStore(bool, &self.should_stop, true, .release);
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
    }

    /// Submit a new inference request (called from HTTP fibers, thread-safe).
    ///
    /// Returns a *seq_slot* index that the caller can use with `getStatus`
    /// to poll for generated tokens and completion. Returns `null` if the
    /// pending request buffer is full (caller should retry or return 429).
    ///
    /// The request is not immediately visible to the scheduler — it will be
    /// drained on the next iteration.
    pub fn submitRequest(self: *Scheduler, req: PendingRequest) ?u32 {
        // Spin to acquire the mutex. This is fine — contention is rare
        // (HTTP fibers submit at ~1000 req/s, the scheduler drains once
        // per iteration) and the critical section is tiny.
        while (!self.request_mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.request_mutex.unlock();

        if (self.pending_count >= MAX_PENDING_REQUESTS) return null;

        // Find a free slot for this sequence. We scan from next_slot_hint
        // to avoid O(n) scans on every call.
        const slot = self.findFreeSlot() orelse return null;

        self.pending_requests[self.pending_count] = req;
        self.pending_count += 1;

        // Reserve the slot. We write a sentinel (max u64) to mark it as
        // "reserved but not yet assigned a seq_id". The scheduler thread
        // will overwrite this with the real seq_id when it drains the
        // request.
        self.seq_id_to_slot[slot] = std.math.maxInt(u64);

        self.seq_status[slot] = .{};
        self.pending_slots[self.pending_count - 1] = slot;

        return slot;
    }

    /// Get the status of a sequence by its slot index.
    ///
    /// The returned pointer is stable for the lifetime of the scheduler.
    /// The caller reads fields using atomic loads (or plain reads if they
    /// can tolerate slight staleness).
    pub fn getStatus(self: *Scheduler, seq_slot: u32) *SequenceStatus {
        return &self.seq_status[seq_slot];
    }

    // -- Internal: scheduler thread entry point --

    fn schedulerLoop(self: *Scheduler) void {
        while (!@atomicLoad(bool, &self.should_stop, .acquire)) {
            self.drainCompletions();
            self.drainRequests();
            self.computeAndPostSchedule();
            self.iteration += 1;

            // Brief spin hint to avoid burning 100% CPU when idle.
            // In production, this would be replaced with an eventfd wait
            // or a short futex-based sleep when no work is pending.
            std.atomic.spinLoopHint();
        }
    }

    // -- Internal: drain completions from GPU worker --

    /// Read all available completions from the completion ring and update
    /// sequence state accordingly.
    fn drainCompletions(self: *Scheduler) void {
        var completion: CompletionT = undefined;
        while (self.completion_ring.tryPop(&completion)) {
            const seq = self.sequences.get(completion.seq_id) orelse continue;
            seq.generated_tokens += 1;

            // Find the slot for this sequence to update the status.
            const slot = self.slotForSeqId(completion.seq_id);

            // Check if the sequence is done.
            const is_eos = completion.is_eos == 1;
            if (is_eos or seq.isAtLimit()) {
                seq.state = .finished;
                self.freeSequenceBlocks(completion.seq_id, slot);
                if (slot) |s| {
                    // Store is_done BEFORE tokens_generated. The HTTP fiber
                    // reads tokens_generated with .acquire first — if it sees
                    // the new count, the acquire guarantees it also sees
                    // is_done=1 (which was stored with .release before the
                    // tokens_generated store below).
                    @atomicStore(u8, &self.seq_status[s].is_done, 1, .release);
                }
            }

            if (slot) |s| {
                @atomicStore(u32, &self.seq_status[s].last_token_id, completion.token_id, .release);
                // This is the "publishing" store — the HTTP fiber's .acquire
                // load on tokens_generated synchronizes with this .release.
                @atomicStore(u32, &self.seq_status[s].tokens_generated, seq.generated_tokens, .release);
            }
        }
    }

    // -- Internal: drain pending requests --

    /// Move all pending requests from the mutex-protected buffer into the
    /// sequence table.
    fn drainRequests(self: *Scheduler) void {
        // Snapshot pending requests under the mutex, then process outside
        // to minimize lock hold time.
        var local_requests: [MAX_PENDING_REQUESTS]PendingRequest = undefined;
        var local_slots: [MAX_PENDING_REQUESTS]u32 = undefined;
        var count: u32 = 0;

        {
            while (!self.request_mutex.tryLock()) {
                std.atomic.spinLoopHint();
            }
            defer self.request_mutex.unlock();

            count = self.pending_count;
            if (count > 0) {
                @memcpy(local_requests[0..count], self.pending_requests[0..count]);
                @memcpy(local_slots[0..count], self.pending_slots[0..count]);
                self.pending_count = 0;
            }
        }

        // Add each request to the sequence table.
        for (0..count) |i| {
            const req = local_requests[i];
            const slot = local_slots[i];

            const seq_id = self.sequences.add(
                req.num_prompt_tokens,
                req.max_tokens,
                req.temperature,
                req.top_p,
                req.priority,
            );

            if (seq_id) |id| {
                // Record the mapping.
                self.seq_id_to_slot[slot] = id;

                // Set arrival iteration on the sequence.
                if (self.sequences.get(id)) |seq| {
                    seq.arrival_iteration = @intCast(self.iteration);
                }

                // Update the status entry with the assigned seq_id.
                @atomicStore(u64, &self.seq_status[slot].seq_id, id, .release);
            } else {
                // Sequence table is full — release the reserved slot.
                self.seq_id_to_slot[slot] = 0;
                @atomicStore(u8, &self.seq_status[slot].is_done, 1, .release);
            }
        }
    }

    // -- Internal: compute and post schedule --

    /// Build a ScheduleT for this iteration and push it to the schedule ring.
    fn computeAndPostSchedule(self: *Scheduler) void {
        var schedule = std.mem.zeroes(ScheduleT);
        schedule.iteration_id = self.iteration;

        var batch_idx: u32 = 0;

        // Step 1: Continue all running sequences.
        {
            var iter = self.sequences.iterByState(.running);
            while (iter.next()) |seq| {
                if (batch_idx >= MAX_BATCH) break;

                // Check if the sequence needs a new block. A sequence needs
                // a new block when its total tokens have outgrown the blocks
                // it currently has.
                const total_tokens = seq.totalTokens();
                const blocks_needed = (total_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
                if (blocks_needed > seq.num_blocks) {
                    // Allocate one new block.
                    if (self.block_alloc.alloc()) |new_block| {
                        const slot = self.slotForSeqId(seq.seq_id);
                        if (slot) |s| {
                            self.block_table.set(s, seq.num_blocks, new_block);
                        }
                        seq.num_blocks = blocks_needed;
                    }
                    // If allocation fails, the sequence continues with its
                    // current blocks — it will retry next iteration.
                }

                self.fillScheduleSlot(&schedule, batch_idx, seq);
                batch_idx += 1;
            }
        }

        // Step 2: Promote waiting sequences if blocks are available.
        {
            var iter = self.sequences.iterByState(.waiting);
            while (iter.next()) |seq| {
                if (batch_idx >= MAX_BATCH) break;

                // Allocate initial blocks for the prompt.
                const blocks_needed = (seq.prompt_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
                if (blocks_needed > self.block_alloc.available()) {
                    // Not enough free blocks — skip this sequence.
                    continue;
                }

                // Allocate all needed blocks.
                const slot = self.slotForSeqId(seq.seq_id);
                var allocated: u32 = 0;
                var alloc_ok = true;
                for (0..blocks_needed) |j| {
                    if (self.block_alloc.alloc()) |new_block| {
                        if (slot) |s| {
                            self.block_table.set(s, @intCast(j), new_block);
                        }
                        allocated += 1;
                    } else {
                        alloc_ok = false;
                        break;
                    }
                }

                if (!alloc_ok) {
                    // Partial allocation — free what we got and skip.
                    if (slot) |s| {
                        for (0..allocated) |j| {
                            const blk = self.block_table.get(s, @intCast(j));
                            self.block_alloc.free(blk);
                        }
                    }
                    continue;
                }

                seq.num_blocks = blocks_needed;
                seq.state = .running;
                seq.last_run_iteration = @intCast(self.iteration);

                self.fillScheduleSlot(&schedule, batch_idx, seq);
                schedule.is_prefill[batch_idx] = 1; // First iteration is prefill.
                batch_idx += 1;
            }
        }

        schedule.num_sequences = batch_idx;

        // Only post if there's work to do.
        if (batch_idx > 0) {
            // Fence block table writes before posting the schedule.
            self.block_table.fence();
            _ = self.schedule_ring.tryPush(&schedule);
        }
    }

    /// Fill a single slot in the ScheduleT with metadata from a sequence.
    fn fillScheduleSlot(self: *Scheduler, schedule: *ScheduleT, idx: u32, seq: *Sequence) void {
        schedule.seq_ids[idx] = seq.seq_id;
        schedule.positions[idx] = seq.totalTokens();
        schedule.seq_lens[idx] = seq.totalTokens();
        schedule.temperatures[idx] = seq.temperature;
        schedule.top_ps[idx] = seq.top_p;
        schedule.max_tokens[idx] = seq.max_tokens;
        schedule.num_blocks[idx] = seq.num_blocks;

        // Copy block table entries for this sequence into the schedule.
        const slot = self.slotForSeqId(seq.seq_id);
        if (slot) |s| {
            const blocks = self.block_table.seqBlocks(s);
            for (0..seq.num_blocks) |j| {
                schedule.block_tables[idx][j] = blocks[j];
            }
        }

        seq.last_run_iteration = @intCast(self.iteration);
    }

    // -- Internal: block cleanup --

    /// Free all KV-cache blocks owned by a sequence and clean up its slot.
    fn freeSequenceBlocks(self: *Scheduler, seq_id: u64, slot: ?u32) void {
        const seq = self.sequences.get(seq_id) orelse return;

        if (slot) |s| {
            for (0..seq.num_blocks) |j| {
                const blk = self.block_table.get(s, @intCast(j));
                self.block_alloc.free(blk);
            }
            self.block_table.clearSeq(s);
            // Release the slot.
            self.seq_id_to_slot[s] = 0;
        }

        // Remove from the sequence table.
        _ = self.sequences.remove(seq_id);
    }

    // -- Internal: slot management --

    /// Find a free slot in the seq_id_to_slot map.
    fn findFreeSlot(self: *Scheduler) ?u32 {
        var i: u32 = 0;
        while (i < self.max_seq_slots) : (i += 1) {
            const idx = (self.next_slot_hint + i) % self.max_seq_slots;
            if (self.seq_id_to_slot[idx] == 0) {
                self.next_slot_hint = (idx + 1) % self.max_seq_slots;
                return idx;
            }
        }
        return null;
    }

    /// Look up the slot index for a given seq_id. Linear scan — the
    /// sequence table is small enough that this is faster than a hash map.
    fn slotForSeqId(self: *Scheduler, seq_id: u64) ?u32 {
        for (self.seq_id_to_slot[0..self.max_seq_slots], 0..) |stored_id, i| {
            if (stored_id == seq_id) return @intCast(i);
        }
        return null;
    }

};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Backing storage sizes for test schedulers.
const TEST_SEQ_CAPACITY = 16;
const TEST_BLOCK_COUNT = 64;
const TEST_BLOCKS_PER_SEQ = 32;
const TEST_RING_CAPACITY = 4;

/// All the backing buffers needed to create a test scheduler, bundled so
/// they can be freed as a group.
const TestContext = struct {
    seq_backing: []Sequence,
    free_stack: []u32,
    ref_counts: []u16,
    block_table_buf: []align(4) u8,
    schedule_ring_buf: []align(64) u8,
    completion_ring_buf: []align(64) u8,
    seq_status_buf: []SequenceStatus,
    slot_map_buf: []u64,
    sched: Scheduler,

    fn deinit(self: *TestContext) void {
        const ally = testing.allocator;
        ally.free(self.seq_backing);
        ally.free(self.free_stack);
        ally.free(self.ref_counts);
        ally.free(self.block_table_buf);
        ally.free(self.schedule_ring_buf);
        ally.free(self.completion_ring_buf);
        ally.free(self.seq_status_buf);
        ally.free(self.slot_map_buf);
    }
};

fn makeTestScheduler() !TestContext {
    const ally = testing.allocator;

    const seq_backing = try ally.alloc(Sequence, TEST_SEQ_CAPACITY);
    const free_stack = try ally.alloc(u32, TEST_BLOCK_COUNT);
    const ref_counts = try ally.alloc(u16, TEST_BLOCK_COUNT);

    const bt_size = BlockTable.totalSize(TEST_SEQ_CAPACITY, TEST_BLOCKS_PER_SEQ);
    const block_table_buf = try ally.alignedAlloc(u8, .@"4", bt_size);
    @memset(block_table_buf, 0);

    const sched_ring_size = SpscRing(ScheduleT).totalSize(TEST_RING_CAPACITY);
    const schedule_ring_buf = try ally.alignedAlloc(u8, .@"64", sched_ring_size);
    @memset(schedule_ring_buf, 0);

    const comp_ring_size = SpscRing(CompletionT).totalSize(TEST_RING_CAPACITY);
    const completion_ring_buf = try ally.alignedAlloc(u8, .@"64", comp_ring_size);
    @memset(completion_ring_buf, 0);

    const seq_status_buf = try ally.alloc(SequenceStatus, TEST_SEQ_CAPACITY);
    const slot_map_buf = try ally.alloc(u64, TEST_SEQ_CAPACITY);

    const sched = Scheduler.init(
        seq_backing,
        free_stack,
        ref_counts,
        TEST_BLOCK_COUNT,
        block_table_buf.ptr,
        0,
        TEST_BLOCKS_PER_SEQ,
        schedule_ring_buf.ptr,
        0,
        TEST_RING_CAPACITY,
        completion_ring_buf.ptr,
        0,
        TEST_RING_CAPACITY,
        seq_status_buf,
        slot_map_buf,
    );

    return .{
        .seq_backing = seq_backing,
        .free_stack = free_stack,
        .ref_counts = ref_counts,
        .block_table_buf = block_table_buf,
        .schedule_ring_buf = schedule_ring_buf,
        .completion_ring_buf = completion_ring_buf,
        .seq_status_buf = seq_status_buf,
        .slot_map_buf = slot_map_buf,
        .sched = sched,
    };
}

test "init and basic state" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    try testing.expectEqual(@as(u64, 0), ctx.sched.iteration);
    try testing.expectEqual(@as(u32, 0), ctx.sched.sequences.len());
    try testing.expectEqual(@as(u32, TEST_BLOCK_COUNT), ctx.sched.block_alloc.available());
    try testing.expectEqual(false, @atomicLoad(bool, &ctx.sched.should_stop, .acquire));
}

test "submitRequest returns a slot and populates pending buffer" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    const req = PendingRequest{
        .num_prompt_tokens = 32,
        .max_tokens = 100,
        .temperature = @as(f16, 0.7),
        .top_p = @as(f16, 0.9),
        .priority = 0,
    };

    const slot = ctx.sched.submitRequest(req);
    try testing.expect(slot != null);
    try testing.expectEqual(@as(u32, 1), ctx.sched.pending_count);
}

test "submitRequest returns null when all slots are used" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    const req = PendingRequest{
        .num_prompt_tokens = 32,
        .max_tokens = 100,
        .temperature = @as(f16, 0.7),
        .top_p = @as(f16, 0.9),
        .priority = 0,
    };

    // Fill up all sequence slots (TEST_SEQ_CAPACITY = 16).
    for (0..TEST_SEQ_CAPACITY) |_| {
        try testing.expect(ctx.sched.submitRequest(req) != null);
    }

    // Next submit should fail because all slots are reserved.
    try testing.expect(ctx.sched.submitRequest(req) == null);
}

test "drainRequests moves pending requests into sequence table" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    const req = PendingRequest{
        .num_prompt_tokens = 32,
        .max_tokens = 100,
        .temperature = @as(f16, 0.7),
        .top_p = @as(f16, 0.9),
        .priority = 0,
    };

    const slot = ctx.sched.submitRequest(req).?;

    // Pending buffer has 1 item.
    try testing.expectEqual(@as(u32, 1), ctx.sched.pending_count);

    // Drain.
    ctx.sched.drainRequests();

    // Pending buffer is now empty.
    try testing.expectEqual(@as(u32, 0), ctx.sched.pending_count);

    // Sequence table has 1 entry.
    try testing.expectEqual(@as(u32, 1), ctx.sched.sequences.len());

    // The seq_id was recorded in the status entry.
    const status = ctx.sched.getStatus(slot);
    const seq_id = @atomicLoad(u64, &status.seq_id, .acquire);
    try testing.expect(seq_id > 0);

    // The sequence is in the waiting state.
    const seq = ctx.sched.sequences.get(seq_id).?;
    try testing.expectEqual(SequenceState.waiting, seq.state);
    try testing.expectEqual(@as(u32, 32), seq.prompt_tokens);
    try testing.expectEqual(@as(u32, 100), seq.max_tokens);
}

test "computeAndPostSchedule promotes waiting sequences" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    // Submit a request.
    const req = PendingRequest{
        .num_prompt_tokens = 32,
        .max_tokens = 100,
        .temperature = @as(f16, 0.7),
        .top_p = @as(f16, 0.9),
        .priority = 0,
    };
    const slot = ctx.sched.submitRequest(req).?;
    ctx.sched.drainRequests();

    // Compute schedule — should promote the waiting sequence.
    ctx.sched.computeAndPostSchedule();

    // The sequence should now be running.
    const seq_id = @atomicLoad(u64, &ctx.sched.seq_status[slot].seq_id, .acquire);
    const seq = ctx.sched.sequences.get(seq_id).?;
    try testing.expectEqual(SequenceState.running, seq.state);

    // Blocks should have been allocated: ceil(32 / 16) = 2 blocks.
    try testing.expectEqual(@as(u32, 2), seq.num_blocks);

    // A schedule should have been pushed to the ring.
    var sched_out: ScheduleT = undefined;
    try testing.expect(ctx.sched.schedule_ring.tryPop(&sched_out));
    try testing.expectEqual(@as(u32, 1), sched_out.num_sequences);
    try testing.expectEqual(seq_id, sched_out.seq_ids[0]);
    try testing.expectEqual(@as(u8, 1), sched_out.is_prefill[0]);
}

test "drainCompletions updates sequence state" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    // Submit and drain a request, then schedule it.
    const req = PendingRequest{
        .num_prompt_tokens = 16,
        .max_tokens = 10,
        .temperature = @as(f16, 1.0),
        .top_p = @as(f16, 1.0),
        .priority = 0,
    };
    const slot = ctx.sched.submitRequest(req).?;
    ctx.sched.drainRequests();
    ctx.sched.computeAndPostSchedule();

    // Pop the schedule so the ring has space.
    var sched_out: ScheduleT = undefined;
    _ = ctx.sched.schedule_ring.tryPop(&sched_out);

    // Simulate the GPU worker sending a completion.
    const seq_id = @atomicLoad(u64, &ctx.sched.seq_status[slot].seq_id, .acquire);
    var completion = CompletionT{
        .seq_id = seq_id,
        .token_id = 42,
        .logprob = @as(f16, -1.0),
        .is_eos = 0,
    };
    try testing.expect(ctx.sched.completion_ring.tryPush(&completion));

    // Drain completions.
    ctx.sched.drainCompletions();

    // Check that the sequence's generated_tokens incremented.
    const seq = ctx.sched.sequences.get(seq_id).?;
    try testing.expectEqual(@as(u32, 1), seq.generated_tokens);

    // Check the status was updated.
    try testing.expectEqual(@as(u32, 42), @atomicLoad(u32, &ctx.sched.seq_status[slot].last_token_id, .acquire));
    try testing.expectEqual(@as(u32, 1), @atomicLoad(u32, &ctx.sched.seq_status[slot].tokens_generated, .acquire));
    try testing.expectEqual(@as(u8, 0), @atomicLoad(u8, &ctx.sched.seq_status[slot].is_done, .acquire));
}

test "sequence finishes on EOS" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    const req = PendingRequest{
        .num_prompt_tokens = 16,
        .max_tokens = 100,
        .temperature = @as(f16, 1.0),
        .top_p = @as(f16, 1.0),
        .priority = 0,
    };
    const slot = ctx.sched.submitRequest(req).?;
    ctx.sched.drainRequests();
    ctx.sched.computeAndPostSchedule();

    var sched_out: ScheduleT = undefined;
    _ = ctx.sched.schedule_ring.tryPop(&sched_out);

    const seq_id = @atomicLoad(u64, &ctx.sched.seq_status[slot].seq_id, .acquire);
    const blocks_before = ctx.sched.block_alloc.available();

    // Send EOS completion.
    var completion = CompletionT{
        .seq_id = seq_id,
        .token_id = 0,
        .logprob = @as(f16, 0.0),
        .is_eos = 1,
    };
    try testing.expect(ctx.sched.completion_ring.tryPush(&completion));
    ctx.sched.drainCompletions();

    // Sequence should be removed.
    try testing.expect(ctx.sched.sequences.get(seq_id) == null);

    // Blocks should have been freed. We allocated ceil(16/16) = 1 block.
    try testing.expectEqual(blocks_before + 1, ctx.sched.block_alloc.available());

    // Status should show done.
    try testing.expectEqual(@as(u8, 1), @atomicLoad(u8, &ctx.sched.seq_status[slot].is_done, .acquire));

    // Slot should be released.
    try testing.expectEqual(@as(u64, 0), ctx.sched.seq_id_to_slot[slot]);
}

test "multiple sequences scheduled together" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    // Submit 3 requests.
    var slots: [3]u32 = undefined;
    for (0..3) |i| {
        const req = PendingRequest{
            .num_prompt_tokens = 16,
            .max_tokens = 50,
            .temperature = @as(f16, 0.8),
            .top_p = @as(f16, 0.95),
            .priority = 0,
        };
        slots[i] = ctx.sched.submitRequest(req).?;
    }

    ctx.sched.drainRequests();
    try testing.expectEqual(@as(u32, 3), ctx.sched.sequences.len());

    ctx.sched.computeAndPostSchedule();

    // All 3 should be in the schedule.
    var sched_out: ScheduleT = undefined;
    try testing.expect(ctx.sched.schedule_ring.tryPop(&sched_out));
    try testing.expectEqual(@as(u32, 3), sched_out.num_sequences);

    // All sequences should now be running.
    for (slots) |slot| {
        const seq_id = @atomicLoad(u64, &ctx.sched.seq_status[slot].seq_id, .acquire);
        const seq = ctx.sched.sequences.get(seq_id).?;
        try testing.expectEqual(SequenceState.running, seq.state);
    }
}

test "start and stop scheduler thread" {
    var ctx = try makeTestScheduler();
    defer ctx.deinit();

    try ctx.sched.start();
    try testing.expect(ctx.sched.thread != null);

    // Spin-wait until the scheduler has run at least a few iterations.
    // The scheduler loop is very tight (just spin hints when idle), so
    // this should complete almost instantly.
    var spins: u32 = 0;
    while (ctx.sched.iteration < 10 and spins < 1_000_000) : (spins += 1) {
        std.atomic.spinLoopHint();
    }

    ctx.sched.stop();
    try testing.expect(ctx.sched.thread == null);

    // Iteration counter should have advanced.
    try testing.expect(ctx.sched.iteration > 0);
}
