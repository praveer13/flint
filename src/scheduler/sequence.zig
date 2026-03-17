//! Sequence state tracking for the Flint scheduler.
//!
//! Each incoming inference request becomes a **Sequence** that progresses
//! through a well-defined lifecycle:
//!
//!   1. **waiting** -- the request has been admitted and tokenized, but has not
//!      yet been scheduled to the GPU. It sits in the waiting queue until the
//!      scheduler picks it up.
//!
//!   2. **running** -- the scheduler has assigned KV-cache blocks and included
//!      the sequence in the current iteration's `ScheduleT`. During its first
//!      iteration the GPU performs *prefill* (processing the full prompt in one
//!      forward pass); on subsequent iterations it performs *decode* (generating
//!      one token at a time).
//!
//!   3. **preempted** -- GPU memory pressure forced the scheduler to swap this
//!      sequence's KV-cache blocks to CPU (or NVMe) to make room for higher-
//!      priority work. The sequence is still alive and will resume when blocks
//!      become available again.
//!
//!   4. **finished** -- the model emitted an end-of-sequence token, or the
//!      sequence reached its `max_tokens` limit. Its KV-cache blocks have been
//!      freed and the HTTP connection has been closed with `[DONE]`.
//!
//! ## Why 64 bytes?
//!
//! The `Sequence` struct is exactly 64 bytes -- one cache line on all modern
//! x86 and ARM server CPUs. The scheduler's hot loop iterates the entire
//! `SequenceTable` every iteration (~10-50 ms), reading and writing sequence
//! metadata. When each sequence is cache-line-aligned and cache-line-sized:
//!
//!   - No false sharing: two sequences never share a cache line, so updating
//!     one never invalidates the other in another core's L1.
//!   - Predictable prefetching: sequential iteration over the table triggers
//!     the hardware prefetcher, keeping the data path ahead of the CPU.
//!   - L1 residency: 1000 sequences = 64 KB, which fits comfortably in a
//!     typical 48-64 KB L1d cache.
//!
//! ## SequenceTable and swap-removal
//!
//! The `SequenceTable` stores sequences in a flat, pre-allocated array with no
//! heap allocation. Removal uses the *swap-with-last* pattern: the sequence to
//! be removed is overwritten with the last active sequence, and the count is
//! decremented. This gives O(1) removal at the cost of unstable ordering --
//! which is fine because the scheduler does not depend on insertion order (it
//! uses `arrival_iteration` for age-based priority instead).

const std = @import("std");

// ---------------------------------------------------------------------------
// SequenceState
// ---------------------------------------------------------------------------

/// The lifecycle state of an inference sequence.
///
/// Transitions:
///   waiting -> running    (scheduler promotes from queue)
///   running -> preempted  (GPU memory pressure, blocks swapped out)
///   running -> finished   (EOS token or max_tokens reached)
///   preempted -> running  (blocks swapped back in, rescheduled)
pub const SequenceState = enum(u8) {
    /// In queue, not yet scheduled to the GPU.
    waiting,
    /// Currently executing on the GPU (prefill or decode phase).
    running,
    /// Swapped out due to memory pressure, waiting to resume.
    preempted,
    /// Generation complete (EOS or max_tokens). Blocks freed.
    finished,
};

// ---------------------------------------------------------------------------
// Sequence
// ---------------------------------------------------------------------------

/// Per-sequence metadata tracked by the scheduler.
///
/// This is an `extern struct` for two reasons:
///   1. Deterministic, C-ABI-compatible layout with no compiler-inserted
///      padding surprises. Every field sits at a known offset.
///   2. Potential future sharing across the shared-memory boundary (e.g., for
///      a monitoring sidecar that reads sequence state without IPC).
///
/// Total size: exactly 64 bytes (one cache line). See module-level docs for
/// why this matters.
pub const Sequence = extern struct {
    /// Unique identifier assigned when the request is admitted. Monotonically
    /// increasing, never reused while the server is running. Used as the key
    /// for lookups and as the `seq_id` in `ScheduleT` / `CompletionT`.
    seq_id: u64,

    /// Current lifecycle state. The scheduler reads this to decide which
    /// sequences to include in the next iteration.
    state: SequenceState,

    /// Padding to align the following u32 field to a 4-byte boundary.
    /// Without this, the C ABI would insert implicit padding anyway -- making
    /// it explicit keeps the Python dtype (if ever needed) in sync.
    _pad0: [3]u8 = .{ 0, 0, 0 },

    /// Number of tokens in the original prompt. Set once at admission time
    /// and never changes. Used to compute the sequence length for attention
    /// (`prompt_tokens + generated_tokens`) and for prefill scheduling.
    prompt_tokens: u32,

    /// Tokens generated so far. Incremented by one each time the scheduler
    /// drains a `CompletionT` for this sequence from the completion ring.
    generated_tokens: u32,

    /// Maximum tokens to generate. When `generated_tokens >= max_tokens`,
    /// the scheduler marks the sequence as finished even if the model has
    /// not produced an EOS token. Set from the request's `max_tokens` field.
    max_tokens: u32,

    /// Number of logical KV-cache blocks currently allocated to this
    /// sequence. Grows during generation as new tokens fill blocks. The
    /// block manager maps these logical blocks to physical GPU/CPU block IDs
    /// in the shared-memory block table.
    num_blocks: u32,

    /// Priority class (0 = highest). Used by the scheduling policy to order
    /// sequences when GPU capacity is limited. Maps to API-level priority
    /// tiers (e.g., paid vs. free).
    priority: u8,

    /// Padding to align `arrival_iteration` to a 4-byte boundary.
    _pad1: [3]u8 = .{ 0, 0, 0 },

    /// The scheduler iteration number when this sequence was first added to
    /// the table. Used for age-based fairness: among sequences of equal
    /// priority, older ones are scheduled first.
    arrival_iteration: u32,

    /// The scheduler iteration number when this sequence was last included
    /// in a `ScheduleT` posted to the GPU worker. Used for starvation
    /// detection: if `current_iteration - last_run_iteration` exceeds a
    /// threshold, the sequence gets a priority boost.
    last_run_iteration: u32,

    /// Sampling temperature. Higher values (e.g., 1.0) produce more random
    /// output; lower values (e.g., 0.1) make the model more deterministic.
    /// Copied into `ScheduleT.temperatures` each iteration this sequence runs.
    temperature: f16,

    /// Top-p (nucleus sampling) threshold. The sampler considers only the
    /// smallest set of tokens whose cumulative probability exceeds this
    /// value. 1.0 means "consider all tokens". Copied into
    /// `ScheduleT.top_ps` each iteration.
    top_p: f16,

    /// Rough estimate of how many tokens this sequence will still generate.
    /// Used by the preemption policy: sequences close to completion are less
    /// attractive preemption targets because their blocks will be freed soon
    /// anyway. Initially set to `max_tokens`; updated heuristically as
    /// generation progresses.
    estimated_remaining: u32,

    /// Padding to bring the struct to exactly 64 bytes. The fields above
    /// occupy 48 bytes (verified by the comptime assertion below). 16 bytes
    /// of trailing padding complete the cache line.
    _pad2: [16]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

    /// Returns the total sequence length (prompt + generated tokens). This
    /// is the value written to `ScheduleT.seq_lens` and tells the attention
    /// kernel how many KV-cache entries to attend over.
    pub fn totalTokens(self: *const Sequence) u32 {
        return self.prompt_tokens + self.generated_tokens;
    }

    /// Returns true if this sequence has reached its generation limit.
    pub fn isAtLimit(self: *const Sequence) bool {
        return self.generated_tokens >= self.max_tokens;
    }
};

// ---------------------------------------------------------------------------
// Comptime layout assertions
// ---------------------------------------------------------------------------

comptime {
    const assert = std.debug.assert;

    // Total size must be exactly one 64-byte cache line.
    assert(@sizeOf(Sequence) == 64);

    // Verify field offsets to catch layout drift. If any of these fire after
    // a field change, update both the offsets here AND any Python dtype that
    // mirrors this struct.
    assert(@offsetOf(Sequence, "seq_id") == 0);
    assert(@offsetOf(Sequence, "state") == 8);
    assert(@offsetOf(Sequence, "_pad0") == 9);
    assert(@offsetOf(Sequence, "prompt_tokens") == 12);
    assert(@offsetOf(Sequence, "generated_tokens") == 16);
    assert(@offsetOf(Sequence, "max_tokens") == 20);
    assert(@offsetOf(Sequence, "num_blocks") == 24);
    assert(@offsetOf(Sequence, "priority") == 28);
    assert(@offsetOf(Sequence, "_pad1") == 29);
    assert(@offsetOf(Sequence, "arrival_iteration") == 32);
    assert(@offsetOf(Sequence, "last_run_iteration") == 36);
    assert(@offsetOf(Sequence, "temperature") == 40);
    assert(@offsetOf(Sequence, "top_p") == 42);
    assert(@offsetOf(Sequence, "estimated_remaining") == 44);
    assert(@offsetOf(Sequence, "_pad2") == 48);

    // Alignment must be 8 (from the leading u64 seq_id).
    assert(@alignOf(Sequence) == 8);
}

// ---------------------------------------------------------------------------
// SequenceTable
// ---------------------------------------------------------------------------

/// A flat, pre-allocated table of active sequences.
///
/// The scheduler iterates this table every iteration to decide which sequences
/// to run, preempt, or finish. There is no heap allocation and no hash map --
/// just a contiguous array of `Sequence` structs sized to fit in L1 cache.
///
/// ## Capacity
///
/// The backing slice is provided at init time (typically carved from a
/// fixed-buffer allocator or a static array). The table cannot grow beyond
/// this capacity. When it is full, `add()` returns `null` and the admission
/// layer rejects new requests with HTTP 429.
///
/// ## Lookup by seq_id
///
/// `get()` performs a linear scan. With at most a few thousand sequences and
/// the array fitting in L1, a linear scan is faster than a hash-map lookup
/// because it avoids indirection and branch misprediction. If profiling ever
/// shows this is a bottleneck, a small auxiliary index can be added without
/// changing the data layout.
pub const SequenceTable = struct {
    /// Pre-allocated backing array. All entries in `sequences[0..count]` are
    /// active; entries beyond `count` are garbage.
    sequences: []Sequence,

    /// Number of active sequences. Always `<= sequences.len`.
    count: u32,

    /// Monotonically increasing ID generator. Every call to `add()`
    /// increments this and assigns the new value as the sequence's `seq_id`.
    /// IDs are never reused (the u64 space is effectively infinite).
    next_seq_id: u64,

    /// Create a table backed by the given slice. The slice is not zeroed --
    /// only the first `count` (initially 0) entries are considered valid.
    pub fn init(backing: []Sequence) SequenceTable {
        return .{
            .sequences = backing,
            .count = 0,
            .next_seq_id = 1,
        };
    }

    /// Add a new sequence in the `waiting` state. Returns the assigned
    /// `seq_id`, or `null` if the table is full.
    ///
    /// The new sequence is appended at `sequences[count]` and `count` is
    /// incremented. This is O(1).
    pub fn add(
        self: *SequenceTable,
        prompt_tokens: u32,
        max_tokens: u32,
        temperature: f16,
        top_p: f16,
        priority: u8,
    ) ?u64 {
        if (self.count >= self.sequences.len) return null;

        const seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        self.sequences[self.count] = .{
            .seq_id = seq_id,
            .state = .waiting,
            .prompt_tokens = prompt_tokens,
            .generated_tokens = 0,
            .max_tokens = max_tokens,
            .num_blocks = 0,
            .priority = priority,
            .arrival_iteration = 0,
            .last_run_iteration = 0,
            .temperature = temperature,
            .top_p = top_p,
            .estimated_remaining = max_tokens,
        };
        self.count += 1;

        return seq_id;
    }

    /// Remove a sequence by `seq_id`. Returns `true` if the sequence was
    /// found and removed, `false` if no sequence with that ID exists.
    ///
    /// Uses swap-with-last removal: the target entry is overwritten with
    /// the last active entry, then `count` is decremented. This is O(n)
    /// for the scan but O(1) for the actual removal (no shifting). The
    /// ordering of remaining sequences is not preserved, which is fine --
    /// the scheduler uses `arrival_iteration` for ordering, not array
    /// position.
    pub fn remove(self: *SequenceTable, seq_id: u64) bool {
        for (self.sequences[0..self.count], 0..) |*seq, i| {
            if (seq.seq_id == seq_id) {
                self.count -= 1;
                if (i != self.count) {
                    // Swap with last: overwrite the removed entry.
                    self.sequences[i] = self.sequences[self.count];
                }
                return true;
            }
        }
        return false;
    }

    /// Find a sequence by `seq_id`. Returns a pointer into the backing
    /// array, or `null` if not found. The pointer is valid until the next
    /// `remove()` call (which may move entries via swap-removal).
    pub fn get(self: *SequenceTable, seq_id: u64) ?*Sequence {
        for (self.sequences[0..self.count]) |*seq| {
            if (seq.seq_id == seq_id) return seq;
        }
        return null;
    }

    /// Returns an iterator over sequences in the given state. The iterator
    /// yields pointers into the backing array; they are valid until the
    /// next structural modification (`add` / `remove`).
    pub fn iterByState(self: *SequenceTable, state: SequenceState) StateIterator {
        return .{
            .table = self,
            .state = state,
            .index = 0,
        };
    }

    /// Iterator that filters sequences by lifecycle state.
    pub const StateIterator = struct {
        table: *SequenceTable,
        state: SequenceState,
        index: u32,

        /// Advance to the next matching sequence, or return `null` when
        /// exhausted.
        pub fn next(self: *StateIterator) ?*Sequence {
            while (self.index < self.table.count) {
                const idx = self.index;
                self.index += 1;
                if (self.table.sequences[idx].state == self.state) {
                    return &self.table.sequences[idx];
                }
            }
            return null;
        }
    };

    /// Returns the number of active sequences.
    pub fn len(self: *const SequenceTable) u32 {
        return self.count;
    }

    /// Returns true if no sequences are active.
    pub fn isEmpty(self: *const SequenceTable) bool {
        return self.count == 0;
    }

    /// Returns the maximum number of sequences the table can hold.
    pub fn capacity(self: *const SequenceTable) usize {
        return self.sequences.len;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "Sequence is exactly 64 bytes (one cache line)" {
    try testing.expectEqual(@as(usize, 64), @sizeOf(Sequence));
    try testing.expectEqual(@as(usize, 8), @alignOf(Sequence));
}

test "Sequence field offsets match expected layout" {
    try testing.expectEqual(@as(usize, 0), @offsetOf(Sequence, "seq_id"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(Sequence, "state"));
    try testing.expectEqual(@as(usize, 12), @offsetOf(Sequence, "prompt_tokens"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(Sequence, "generated_tokens"));
    try testing.expectEqual(@as(usize, 20), @offsetOf(Sequence, "max_tokens"));
    try testing.expectEqual(@as(usize, 24), @offsetOf(Sequence, "num_blocks"));
    try testing.expectEqual(@as(usize, 28), @offsetOf(Sequence, "priority"));
    try testing.expectEqual(@as(usize, 32), @offsetOf(Sequence, "arrival_iteration"));
    try testing.expectEqual(@as(usize, 36), @offsetOf(Sequence, "last_run_iteration"));
    try testing.expectEqual(@as(usize, 40), @offsetOf(Sequence, "temperature"));
    try testing.expectEqual(@as(usize, 42), @offsetOf(Sequence, "top_p"));
    try testing.expectEqual(@as(usize, 44), @offsetOf(Sequence, "estimated_remaining"));
    try testing.expectEqual(@as(usize, 48), @offsetOf(Sequence, "_pad2"));
}

test "Sequence helper methods" {
    var seq: Sequence = .{
        .seq_id = 1,
        .state = .running,
        .prompt_tokens = 100,
        .generated_tokens = 50,
        .max_tokens = 200,
        .num_blocks = 10,
        .priority = 0,
        .arrival_iteration = 0,
        .last_run_iteration = 5,
        .temperature = @as(f16, 0.8),
        .top_p = @as(f16, 0.95),
        .estimated_remaining = 150,
    };

    try testing.expectEqual(@as(u32, 150), seq.totalTokens());
    try testing.expect(!seq.isAtLimit());

    seq.generated_tokens = 200;
    try testing.expect(seq.isAtLimit());
    try testing.expectEqual(@as(u32, 300), seq.totalTokens());
}

test "SequenceTable add and get" {
    var backing: [8]Sequence = undefined;
    var table = SequenceTable.init(&backing);

    try testing.expectEqual(@as(u32, 0), table.len());
    try testing.expect(table.isEmpty());

    const id1 = table.add(128, 256, @as(f16, 0.7), @as(f16, 0.9), 1).?;
    try testing.expectEqual(@as(u64, 1), id1);
    try testing.expectEqual(@as(u32, 1), table.len());
    try testing.expect(!table.isEmpty());

    const seq = table.get(id1).?;
    try testing.expectEqual(@as(u64, 1), seq.seq_id);
    try testing.expectEqual(SequenceState.waiting, seq.state);
    try testing.expectEqual(@as(u32, 128), seq.prompt_tokens);
    try testing.expectEqual(@as(u32, 0), seq.generated_tokens);
    try testing.expectEqual(@as(u32, 256), seq.max_tokens);
    try testing.expectEqual(@as(u32, 0), seq.num_blocks);
    try testing.expectEqual(@as(u8, 1), seq.priority);
    try testing.expectEqual(@as(f16, 0.7), seq.temperature);
    try testing.expectEqual(@as(f16, 0.9), seq.top_p);
    try testing.expectEqual(@as(u32, 256), seq.estimated_remaining);

    // Non-existent ID returns null.
    try testing.expect(table.get(999) == null);
}

test "SequenceTable remove uses swap-with-last" {
    var backing: [8]Sequence = undefined;
    var table = SequenceTable.init(&backing);

    const id1 = table.add(10, 100, @as(f16, 1.0), @as(f16, 1.0), 0).?;
    const id2 = table.add(20, 200, @as(f16, 1.0), @as(f16, 1.0), 0).?;
    const id3 = table.add(30, 300, @as(f16, 1.0), @as(f16, 1.0), 0).?;
    try testing.expectEqual(@as(u32, 3), table.len());

    // Remove the middle entry (id2). The last entry (id3) should be
    // swapped into its position.
    try testing.expect(table.remove(id2));
    try testing.expectEqual(@as(u32, 2), table.len());

    // id2 is gone.
    try testing.expect(table.get(id2) == null);

    // id1 and id3 are still reachable.
    try testing.expect(table.get(id1) != null);
    try testing.expect(table.get(id3) != null);

    // Removing a non-existent ID returns false and does not change count.
    try testing.expect(!table.remove(id2));
    try testing.expectEqual(@as(u32, 2), table.len());
}

test "SequenceTable full returns null on add" {
    var backing: [2]Sequence = undefined;
    var table = SequenceTable.init(&backing);

    const id1 = table.add(10, 100, @as(f16, 1.0), @as(f16, 1.0), 0);
    const id2 = table.add(20, 200, @as(f16, 1.0), @as(f16, 1.0), 0);
    try testing.expect(id1 != null);
    try testing.expect(id2 != null);
    try testing.expectEqual(@as(u32, 2), table.len());

    // Table is full -- next add returns null.
    const id3 = table.add(30, 300, @as(f16, 1.0), @as(f16, 1.0), 0);
    try testing.expect(id3 == null);
    try testing.expectEqual(@as(u32, 2), table.len());
}

test "SequenceTable iterByState filters correctly" {
    var backing: [8]Sequence = undefined;
    var table = SequenceTable.init(&backing);

    _ = table.add(10, 100, @as(f16, 1.0), @as(f16, 1.0), 0); // id=1, waiting
    _ = table.add(20, 200, @as(f16, 1.0), @as(f16, 1.0), 0); // id=2, waiting
    _ = table.add(30, 300, @as(f16, 1.0), @as(f16, 1.0), 0); // id=3, waiting

    // Transition id=1 and id=3 to running.
    table.get(1).?.state = .running;
    table.get(3).?.state = .running;

    // Iterate waiting -- should yield only id=2.
    {
        var iter = table.iterByState(.waiting);
        const first = iter.next().?;
        try testing.expectEqual(@as(u64, 2), first.seq_id);
        try testing.expect(iter.next() == null);
    }

    // Iterate running -- should yield id=1 and id=3 (order depends on
    // array position, not insertion order after swaps).
    {
        var iter = table.iterByState(.running);
        var count: u32 = 0;
        while (iter.next()) |_| count += 1;
        try testing.expectEqual(@as(u32, 2), count);
    }

    // Iterate preempted -- should yield nothing.
    {
        var iter = table.iterByState(.preempted);
        try testing.expect(iter.next() == null);
    }
}

test "SequenceTable state transitions" {
    var backing: [4]Sequence = undefined;
    var table = SequenceTable.init(&backing);

    const id = table.add(50, 100, @as(f16, 0.5), @as(f16, 0.9), 2).?;

    // Initial state is waiting.
    const seq = table.get(id).?;
    try testing.expectEqual(SequenceState.waiting, seq.state);

    // Transition to running (scheduler promotes from queue).
    seq.state = .running;
    try testing.expectEqual(SequenceState.running, table.get(id).?.state);

    // Transition to preempted (memory pressure).
    seq.state = .preempted;
    try testing.expectEqual(SequenceState.preempted, table.get(id).?.state);

    // Resume: back to running.
    seq.state = .running;
    try testing.expectEqual(SequenceState.running, table.get(id).?.state);

    // Finish.
    seq.state = .finished;
    try testing.expectEqual(SequenceState.finished, table.get(id).?.state);
}

test "SequenceTable seq_id is monotonically increasing" {
    var backing: [8]Sequence = undefined;
    var table = SequenceTable.init(&backing);

    var prev_id: u64 = 0;
    for (0..5) |_| {
        const id = table.add(10, 100, @as(f16, 1.0), @as(f16, 1.0), 0).?;
        try testing.expect(id > prev_id);
        prev_id = id;
    }

    // Remove some and add more -- IDs must still increase.
    try testing.expect(table.remove(2));
    try testing.expect(table.remove(4));

    const id_after = table.add(10, 100, @as(f16, 1.0), @as(f16, 1.0), 0).?;
    try testing.expect(id_after > prev_id);
}
