"""Flint shared memory bindings for Python GPU workers.

This package provides Python-side access to the shared memory ring buffers
and data structures used for IPC between the Zig server and Python GPU
workers. All types mirror the Zig extern struct layouts byte-for-byte.
"""

from flint_shm.types import (
    MAX_BATCH,
    MAX_BLOCKS_PER_SEQ,
    MAX_PROMPT_LEN,
    MAX_SWAP,
    COMPLETION_DTYPE,
    SWAP_CMD_DTYPE,
    SCHEDULE_DTYPE,
)
from flint_shm.ring_buffer import RingReader, RingWriter, HEADER_SIZE
from flint_shm.heartbeat import HeartbeatRegion

try:
    from flint_shm.tokenizer import Tokenizer
except ImportError:
    pass  # transformers not installed

__all__ = [
    "MAX_BATCH",
    "MAX_BLOCKS_PER_SEQ",
    "MAX_PROMPT_LEN",
    "MAX_SWAP",
    "COMPLETION_DTYPE",
    "SWAP_CMD_DTYPE",
    "SCHEDULE_DTYPE",
    "RingReader",
    "RingWriter",
    "HEADER_SIZE",
    "HeartbeatRegion",
    "Tokenizer",
]
