//! Shared memory packed struct types — the binary ABI between Zig and Python.
//!
//! These types define the exact byte layout of every structure that crosses the
//! shared memory boundary between the Zig server process and the Python GPU
//! worker processes. Both sides interpret the same raw bytes: Zig through these
//! `extern struct` definitions, Python through matching `numpy.dtype` layouts.
//!
//! Why `extern struct`?
//!   Zig's default struct layout is allowed to reorder fields and insert
//!   padding for performance. `extern struct` forces C ABI layout: fields
//!   appear in declaration order with alignment-dictated padding. This gives
//!   us a deterministic byte layout that we can reproduce exactly in Python
//!   (and in any other language that understands C struct layout rules).
//!
//! If you change ANY field in these structs, you MUST update the corresponding
//! numpy dtype in `python/flint_shm/` AND re-run the cross-language roundtrip
//! test. A layout mismatch means one side reads garbage from the other — there
//! is no version negotiation or magic number check at runtime.
//!
//! Background for LLM inference concepts used in these types:
//!
//!   - **KV-cache blocks**: During autoregressive generation, the model stores
//!     intermediate Key and Value tensors for every token processed so far.
//!     This "KV-cache" is divided into fixed-size blocks (e.g., 16 tokens per
//!     block) and managed like OS virtual memory pages. `block_tables` maps
//!     logical block indices to physical GPU memory block IDs.
//!
//!   - **Prefill vs. decode**: Processing a prompt (many tokens at once) is
//!     called "prefill". Generating new tokens one at a time is "decode".
//!     The GPU kernel uses different code paths for each, so the scheduler
//!     must mark which phase each sequence is in.
//!
//!   - **Sampling parameters**: After the model produces logits (a probability
//!     distribution over the vocabulary), we sample the next token.
//!     `temperature` controls randomness (higher = more random), `top_p`
//!     (nucleus sampling) limits the sample to the smallest set of tokens
//!     whose cumulative probability exceeds `top_p`.

const std = @import("std");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of sequences that can be scheduled in a single iteration.
/// This bounds the size of per-sequence arrays in `ScheduleT`. 256 is a
/// typical upper bound for GPU batch sizes on current hardware.
pub const MAX_BATCH: u32 = 256;

/// Maximum number of KV-cache blocks that a single sequence can occupy.
/// With 16 tokens per block and 512 blocks, this supports sequences up to
/// 8192 tokens long.
pub const MAX_BLOCKS_PER_SEQ: u32 = 512;

/// Maximum prompt length in tokens. Prompts longer than this are rejected
/// at admission time.
pub const MAX_PROMPT_LEN: u32 = 8192;

/// Maximum number of KV-cache block swap commands (GPU <-> CPU) per
/// iteration. Swaps are expensive, so this is kept modest.
pub const MAX_SWAP: u32 = 64;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single output token produced by the GPU worker.
///
/// The Python worker writes one `CompletionT` per generated token into the
/// completion ring buffer. The Zig scheduler reads these to update sequence
/// state (increment token count, detect end-of-sequence, record log
/// probabilities for quality metrics).
///
/// Size: 16 bytes. This is a power of two, which keeps ring buffer slot
/// arithmetic simple and avoids straddling cache lines at common ring sizes.
pub const CompletionT = extern struct {
    /// Unique identifier for the sequence this token belongs to. Assigned
    /// by the Zig server when the request arrives; never reused while the
    /// sequence is alive.
    seq_id: u64,

    /// The vocabulary index of the generated token. The tokenizer maps this
    /// back to a text string on the Zig side before sending it to the client.
    token_id: u32,

    /// Log-probability of the chosen token under the model's distribution.
    /// Useful for quality monitoring and for the `logprobs` field in the
    /// OpenAI-compatible API response. Stored as f16 to save space — the
    /// precision loss is negligible for log-probs.
    ///
    /// Placed before `is_eos` so that it falls on a 2-byte-aligned offset
    /// (12) without implicit padding. In C ABI layout, an f16 requires
    /// 2-byte alignment; if it followed the u8 `is_eos` at offset 12, the
    /// compiler would insert 1 byte of hidden padding.
    logprob: f16,

    /// 1 if this token is the end-of-sequence marker (the model has decided
    /// to stop generating), 0 otherwise. When the scheduler sees `is_eos=1`,
    /// it marks the sequence as finished and frees its KV-cache blocks.
    is_eos: u8,

    /// Explicit padding to bring the struct to exactly 16 bytes. The fields
    /// above occupy 15 bytes (u64 + u32 + f16 + u8 = 8 + 4 + 2 + 1). The
    /// struct's alignment is 8 (from the u64), so without this pad the C
    /// ABI would round the size up to 16 anyway — but making the pad
    /// explicit ensures the Python numpy dtype matches without guesswork.
    _pad: [1]u8 = .{0},
};

/// A KV-cache block swap command.
///
/// When GPU memory is full and the scheduler needs to make room for
/// higher-priority sequences, it "swaps out" KV-cache blocks from GPU to
/// CPU memory. Later, when the preempted sequence resumes, those blocks are
/// "swapped in" from CPU back to GPU. Each `SwapCmd` describes one such
/// block transfer.
///
/// Size: 8 bytes (two u32 block IDs, no padding needed).
pub const SwapCmd = extern struct {
    /// The block ID on the GPU side. For swap-out, this is the source; for
    /// swap-in, this is the destination.
    gpu_block_id: u32,

    /// The block ID on the CPU side. For swap-out, this is the destination;
    /// for swap-in, this is the source.
    cpu_block_id: u32,
};

/// One iteration's worth of work posted by the Zig scheduler for the GPU
/// worker to execute.
///
/// Every scheduling iteration (roughly every forward pass, ~10-50ms), the
/// scheduler decides which sequences to run, assembles their metadata into
/// a `ScheduleT`, and pushes it onto the schedule ring buffer. The Python
/// worker pops it, translates it into vLLM's `AttentionMetadata`, and calls
/// `ModelRunner.execute_model()`.
///
/// This is a large struct (several hundred KB) because it carries the full
/// block tables for every sequence in the batch. It lives in a shared-memory
/// ring buffer slot, not on the stack.
pub const ScheduleT = extern struct {
    /// Monotonically increasing iteration counter. Used by both sides to
    /// detect missed iterations and for metrics/logging.
    iteration_id: u64,

    /// Number of active sequences in this iteration. Only the first
    /// `num_sequences` entries in the per-sequence arrays below are valid.
    num_sequences: u32,

    /// Explicit padding so that `seq_ids` (a u64 array) starts at an
    /// 8-byte-aligned offset. Without this, the C ABI would insert 4
    /// bytes of implicit padding between `num_sequences` (offset 12) and
    /// `seq_ids` (which requires 8-byte alignment at offset 16). Making
    /// it explicit keeps the Python dtype in sync.
    _align_pad: [4]u8 = .{ 0, 0, 0, 0 },

    /// Sequence IDs for each slot in this batch.
    seq_ids: [MAX_BATCH]u64,

    /// The next token ID to feed into the model for each sequence. During
    /// decode, this is the token generated in the previous iteration.
    /// During prefill, this field is unused (the full prompt is read from
    /// the request ring).
    token_ids: [MAX_BATCH]u32,

    /// The absolute position of the next token in each sequence. Used by
    /// rotary position embeddings (RoPE) in the model's attention layers.
    positions: [MAX_BATCH]u32,

    /// Current total length of each sequence (prompt + generated tokens so
    /// far). The attention kernel uses this to know how many KV-cache
    /// entries to attend over.
    seq_lens: [MAX_BATCH]u32,

    /// 1 if this sequence is in the prefill phase (processing the prompt),
    /// 0 if it is in the decode phase (generating tokens one at a time).
    /// The GPU kernel dispatches to different code paths for each.
    is_prefill: [MAX_BATCH]u8,

    /// Sampling temperature per sequence. Higher values (e.g. 1.0) produce
    /// more random output; lower values (e.g. 0.1) make the model more
    /// deterministic. Stored as f16 — sufficient precision for this purpose.
    temperatures: [MAX_BATCH]f16,

    /// Top-p (nucleus sampling) threshold per sequence. The sampler
    /// considers only the smallest set of tokens whose cumulative
    /// probability exceeds this value. 1.0 means "consider all tokens".
    top_ps: [MAX_BATCH]f16,

    /// Maximum number of tokens to generate for each sequence. When a
    /// sequence's generated token count reaches this limit, the scheduler
    /// marks it as finished even if the model has not produced an EOS token.
    max_tokens: [MAX_BATCH]u32,

    /// KV-cache block table for each sequence. `block_tables[i][j]` is the
    /// physical GPU memory block ID for logical block `j` of sequence `i`.
    /// Only the first `num_blocks[i]` entries are valid for sequence `i`.
    /// The attention kernel indexes into this table to find where each
    /// sequence's cached Keys and Values live in GPU memory.
    block_tables: [MAX_BATCH][MAX_BLOCKS_PER_SEQ]u32,

    /// Number of KV-cache blocks currently allocated to each sequence.
    num_blocks: [MAX_BATCH]u32,

    /// Number of swap-out commands in this iteration (GPU -> CPU).
    num_swap_out: u32,

    /// Number of swap-in commands in this iteration (CPU -> GPU).
    num_swap_in: u32,

    /// Block swap-out commands: move these blocks from GPU to CPU memory
    /// to free GPU space for higher-priority sequences.
    swap_out: [MAX_SWAP]SwapCmd,

    /// Block swap-in commands: move these blocks from CPU back to GPU
    /// memory because a previously preempted sequence is resuming.
    swap_in: [MAX_SWAP]SwapCmd,

    /// 1 to signal the worker to shut down gracefully. The worker drains
    /// its current iteration, writes any final completions, and exits.
    is_shutdown: u8,

    /// Explicit padding to align the struct's total size to an 8-byte
    /// boundary. This ensures that arrays of `ScheduleT` (such as ring
    /// buffer slot arrays) maintain proper alignment for the u64 fields.
    _pad: [7]u8 = .{ 0, 0, 0, 0, 0, 0, 0 },
};

// ---------------------------------------------------------------------------
// Comptime assertions
// ---------------------------------------------------------------------------
//
// These verify that the struct layouts match our expectations at compile time.
// If a Zig compiler update or a field change alters the layout, the build
// fails immediately rather than producing silent data corruption at runtime.

comptime {
    const assert = std.debug.assert;

    // -- CompletionT: must be exactly 16 bytes --
    assert(@sizeOf(CompletionT) == 16);
    assert(@offsetOf(CompletionT, "seq_id") == 0);
    assert(@offsetOf(CompletionT, "token_id") == 8);
    assert(@offsetOf(CompletionT, "logprob") == 12);
    assert(@offsetOf(CompletionT, "is_eos") == 14);
    assert(@offsetOf(CompletionT, "_pad") == 15);

    // -- SwapCmd: must be exactly 8 bytes --
    assert(@sizeOf(SwapCmd) == 8);
    assert(@offsetOf(SwapCmd, "gpu_block_id") == 0);
    assert(@offsetOf(SwapCmd, "cpu_block_id") == 4);

    // -- ScheduleT: alignment must be 8 (for u64 fields) --
    assert(@alignOf(ScheduleT) == 8);

    // Verify key field offsets to catch layout drift.
    assert(@offsetOf(ScheduleT, "iteration_id") == 0);
    assert(@offsetOf(ScheduleT, "num_sequences") == 8);
    assert(@offsetOf(ScheduleT, "_align_pad") == 12);
    assert(@offsetOf(ScheduleT, "seq_ids") == 16);

    // The total size must be a multiple of 8 (the struct's alignment).
    assert(@sizeOf(ScheduleT) % 8 == 0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "CompletionT field values roundtrip correctly" {
    var c: CompletionT = .{
        .seq_id = 0xDEAD_BEEF_CAFE_BABE,
        .token_id = 31337,
        .logprob = @as(f16, -2.5),
        .is_eos = 1,
    };

    try testing.expectEqual(@as(u64, 0xDEAD_BEEF_CAFE_BABE), c.seq_id);
    try testing.expectEqual(@as(u32, 31337), c.token_id);
    try testing.expectEqual(@as(f16, -2.5), c.logprob);
    try testing.expectEqual(@as(u8, 1), c.is_eos);
    try testing.expectEqual(@as(u8, 0), c._pad[0]);

    // Mutate and re-read to verify writes stick.
    c.seq_id = 42;
    c.is_eos = 0;
    try testing.expectEqual(@as(u64, 42), c.seq_id);
    try testing.expectEqual(@as(u8, 0), c.is_eos);
}

test "CompletionT can be interpreted from raw bytes" {
    // Simulate reading from shared memory: write known bytes, overlay struct.
    // The buffer must be aligned to @alignOf(CompletionT) (8 bytes) because
    // we cast a pointer to it. On the stack, a plain [16]u8 has only 1-byte
    // alignment, which would trip the safety check in @alignCast.
    var bytes: [16]u8 align(@alignOf(CompletionT)) = undefined;
    @memset(&bytes, 0);

    // Write seq_id = 1 at offset 0 (little-endian u64).
    bytes[0] = 1;
    // Write token_id = 256 at offset 8 (little-endian u32).
    bytes[8] = 0;
    bytes[9] = 1;
    // is_eos at offset 14.
    bytes[14] = 1;

    const c: *const CompletionT = @ptrCast(@alignCast(&bytes));
    try testing.expectEqual(@as(u64, 1), c.seq_id);
    try testing.expectEqual(@as(u32, 256), c.token_id);
    try testing.expectEqual(@as(u8, 1), c.is_eos);
}

test "SwapCmd field values roundtrip correctly" {
    const cmd = SwapCmd{
        .gpu_block_id = 100,
        .cpu_block_id = 200,
    };
    try testing.expectEqual(@as(u32, 100), cmd.gpu_block_id);
    try testing.expectEqual(@as(u32, 200), cmd.cpu_block_id);
}

test "ScheduleT can be zero-initialized" {
    // `std.mem.zeroes` is the canonical way to create a blank schedule before
    // filling in the fields for the current iteration.
    const sched = std.mem.zeroes(ScheduleT);

    try testing.expectEqual(@as(u64, 0), sched.iteration_id);
    try testing.expectEqual(@as(u32, 0), sched.num_sequences);
    try testing.expectEqual(@as(u32, 0), sched.num_swap_out);
    try testing.expectEqual(@as(u32, 0), sched.num_swap_in);
    try testing.expectEqual(@as(u8, 0), sched.is_shutdown);
}

test "ScheduleT field access with populated sequences" {
    var sched = std.mem.zeroes(ScheduleT);

    // Simulate scheduling 3 sequences.
    sched.iteration_id = 42;
    sched.num_sequences = 3;

    sched.seq_ids[0] = 100;
    sched.seq_ids[1] = 200;
    sched.seq_ids[2] = 300;

    sched.token_ids[0] = 10;
    sched.token_ids[1] = 20;
    sched.token_ids[2] = 30;

    sched.positions[0] = 50;
    sched.seq_lens[0] = 51;
    sched.is_prefill[0] = 1;
    sched.temperatures[0] = @as(f16, 0.7);
    sched.top_ps[0] = @as(f16, 0.9);
    sched.max_tokens[0] = 1024;

    // Block table for sequence 0: 3 blocks.
    sched.num_blocks[0] = 3;
    sched.block_tables[0][0] = 10;
    sched.block_tables[0][1] = 20;
    sched.block_tables[0][2] = 30;

    // Swap commands.
    sched.num_swap_out = 1;
    sched.swap_out[0] = .{ .gpu_block_id = 5, .cpu_block_id = 50 };

    // Verify reads.
    try testing.expectEqual(@as(u64, 42), sched.iteration_id);
    try testing.expectEqual(@as(u32, 3), sched.num_sequences);
    try testing.expectEqual(@as(u64, 200), sched.seq_ids[1]);
    try testing.expectEqual(@as(u32, 30), sched.token_ids[2]);
    try testing.expectEqual(@as(u32, 50), sched.positions[0]);
    try testing.expectEqual(@as(u32, 51), sched.seq_lens[0]);
    try testing.expectEqual(@as(u8, 1), sched.is_prefill[0]);
    try testing.expectEqual(@as(f16, 0.7), sched.temperatures[0]);
    try testing.expectEqual(@as(f16, 0.9), sched.top_ps[0]);
    try testing.expectEqual(@as(u32, 1024), sched.max_tokens[0]);
    try testing.expectEqual(@as(u32, 3), sched.num_blocks[0]);
    try testing.expectEqual(@as(u32, 20), sched.block_tables[0][1]);
    try testing.expectEqual(@as(u32, 1), sched.num_swap_out);
    try testing.expectEqual(@as(u32, 5), sched.swap_out[0].gpu_block_id);
    try testing.expectEqual(@as(u32, 50), sched.swap_out[0].cpu_block_id);
}

test "ScheduleT shutdown flag" {
    var sched = std.mem.zeroes(ScheduleT);
    sched.is_shutdown = 1;
    try testing.expectEqual(@as(u8, 1), sched.is_shutdown);
}

test "offsetOf for key fields matches expected layout" {
    // CompletionT offsets (verified in comptime block too, but explicit
    // runtime test makes failures easier to diagnose in test output).
    try testing.expectEqual(@as(usize, 0), @offsetOf(CompletionT, "seq_id"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(CompletionT, "token_id"));
    try testing.expectEqual(@as(usize, 12), @offsetOf(CompletionT, "logprob"));
    try testing.expectEqual(@as(usize, 14), @offsetOf(CompletionT, "is_eos"));
    try testing.expectEqual(@as(usize, 15), @offsetOf(CompletionT, "_pad"));

    // SwapCmd offsets.
    try testing.expectEqual(@as(usize, 0), @offsetOf(SwapCmd, "gpu_block_id"));
    try testing.expectEqual(@as(usize, 4), @offsetOf(SwapCmd, "cpu_block_id"));

    // ScheduleT key offsets.
    try testing.expectEqual(@as(usize, 0), @offsetOf(ScheduleT, "iteration_id"));
    try testing.expectEqual(@as(usize, 8), @offsetOf(ScheduleT, "num_sequences"));
    try testing.expectEqual(@as(usize, 16), @offsetOf(ScheduleT, "seq_ids"));
}
