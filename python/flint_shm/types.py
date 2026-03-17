"""Numpy structured dtypes matching Zig extern structs in src/shm/types.zig.

These dtypes define the exact byte layout of every structure that crosses the
shared memory boundary. Field order, sizes, and padding MUST match the Zig
definitions byte-for-byte. If you change anything here, update the Zig side
too and re-run the cross-language roundtrip test.

All multi-byte fields are little-endian ('<' prefix) to match x86-64 native
byte order, which is what Zig's extern structs use on Linux.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants — must match src/shm/types.zig
# ---------------------------------------------------------------------------

MAX_BATCH: int = 256
MAX_BLOCKS_PER_SEQ: int = 512
MAX_PROMPT_LEN: int = 8192
MAX_SWAP: int = 64

# ---------------------------------------------------------------------------
# CompletionT — 16 bytes
# ---------------------------------------------------------------------------
# Zig layout (extern struct, C ABI):
#   offset  0: seq_id    u64
#   offset  8: token_id  u32
#   offset 12: logprob   f16
#   offset 14: is_eos    u8
#   offset 15: _pad      [1]u8
#
# Note: logprob comes BEFORE is_eos (Zig reordered from the original plan
# to avoid implicit padding between a u8 and an f16).

COMPLETION_DTYPE = np.dtype([
    ('seq_id', '<u8'),      # u64 at offset 0
    ('token_id', '<u4'),    # u32 at offset 8
    ('logprob', '<f2'),     # f16 at offset 12
    ('is_eos', 'u1'),       # u8  at offset 14
    ('_pad', 'u1'),         # u8  at offset 15
], align=True)

# ---------------------------------------------------------------------------
# SwapCmd — 8 bytes
# ---------------------------------------------------------------------------
# Zig layout:
#   offset 0: gpu_block_id  u32
#   offset 4: cpu_block_id  u32

SWAP_CMD_DTYPE = np.dtype([
    ('gpu_block_id', '<u4'),    # u32 at offset 0
    ('cpu_block_id', '<u4'),    # u32 at offset 4
], align=True)

# ---------------------------------------------------------------------------
# ScheduleT — large struct (~533792 bytes)
# ---------------------------------------------------------------------------
# Zig layout (extern struct, C ABI):
#   offset      0: iteration_id   u64
#   offset      8: num_sequences  u32
#   offset     12: _align_pad     [4]u8   (explicit padding for u64 alignment)
#   offset     16: seq_ids        [256]u64
#   offset   2064: token_ids      [256]u32
#   offset   3088: positions      [256]u32
#   offset   4112: seq_lens       [256]u32
#   offset   5136: is_prefill     [256]u8
#   offset   5392: temperatures   [256]f16
#   offset   5904: top_ps         [256]f16
#   offset   6416: max_tokens     [256]u32
#   offset   7440: block_tables   [256][512]u32
#   offset 531728: num_blocks     [256]u32
#   offset 532752: num_swap_out   u32
#   offset 532756: num_swap_in    u32
#   offset 532760: swap_out       [64]SwapCmd (64 * 8 = 512 bytes)
#   offset 533272: swap_in        [64]SwapCmd (64 * 8 = 512 bytes)
#   offset 533784: is_shutdown    u8
#   offset 533785: _pad           [7]u8
#   total:  533792 bytes

SCHEDULE_DTYPE = np.dtype([
    ('iteration_id', '<u8'),                            # u64
    ('num_sequences', '<u4'),                           # u32
    ('_align_pad', 'u1', (4,)),                         # [4]u8
    ('seq_ids', '<u8', (MAX_BATCH,)),                   # [256]u64
    ('token_ids', '<u4', (MAX_BATCH,)),                 # [256]u32
    ('positions', '<u4', (MAX_BATCH,)),                 # [256]u32
    ('seq_lens', '<u4', (MAX_BATCH,)),                  # [256]u32
    ('is_prefill', 'u1', (MAX_BATCH,)),                 # [256]u8
    ('temperatures', '<f2', (MAX_BATCH,)),              # [256]f16
    ('top_ps', '<f2', (MAX_BATCH,)),                    # [256]f16
    ('max_tokens', '<u4', (MAX_BATCH,)),                # [256]u32
    ('block_tables', '<u4', (MAX_BATCH, MAX_BLOCKS_PER_SEQ)),  # [256][512]u32
    ('num_blocks', '<u4', (MAX_BATCH,)),                # [256]u32
    ('num_swap_out', '<u4'),                            # u32
    ('num_swap_in', '<u4'),                             # u32
    ('swap_out', SWAP_CMD_DTYPE, (MAX_SWAP,)),          # [64]SwapCmd
    ('swap_in', SWAP_CMD_DTYPE, (MAX_SWAP,)),           # [64]SwapCmd
    ('is_shutdown', 'u1'),                              # u8
    ('_pad', 'u1', (7,)),                               # [7]u8
], align=True)
