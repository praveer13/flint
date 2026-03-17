"""SPSC ring buffer reader/writer over mmap'd shared memory.

Mirrors the Zig SpscRing(T) in src/shm/ring_buffer.zig. The header layout
uses two cache-line-separated u64 counters (head and tail), each padded to
64 bytes to prevent false sharing.

Header layout (128 bytes total):
    offset  0: head (u64) + 56 bytes padding  = 64 bytes (consumer's cache line)
    offset 64: tail (u64) + 56 bytes padding  = 64 bytes (producer's cache line)

Slot data begins immediately after the header at offset 128.

Memory ordering: Python's mmap does not provide the same acquire/release
semantics as Zig's @atomicLoad/@atomicStore. On x86-64, plain loads and
stores have acquire/release semantics by default (Total Store Order), so
struct.unpack_from / struct.pack_into are sufficient for correctness on
the target platform (Linux x86-64). This would NOT be safe on ARM without
explicit barriers.
"""

import mmap
import struct
import time

import numpy as np

# Header occupies two cache lines: head (64 bytes) + tail (64 bytes).
HEADER_SIZE: int = 128
HEAD_OFFSET: int = 0
TAIL_OFFSET: int = 64


class RingReader:
    """Consumer side of an SPSC ring buffer. Reads entries written by Zig.

    The reader advances the head counter after copying data out of a slot.
    Only one thread/process should read from a given ring instance (the
    Single-Consumer guarantee).

    Args:
        mm: An mmap object covering the shared memory region.
        offset: Byte offset within mm where this ring's header starts.
        capacity: Number of slots (must match the producer's capacity).
        dtype: Numpy dtype describing one slot's layout.
    """

    def __init__(self, mm: mmap.mmap, offset: int, capacity: int, dtype: np.dtype):
        self._mm = mm
        self._base = offset
        self._data_offset = offset + HEADER_SIZE
        self._capacity = capacity
        self._dtype = dtype
        self._slot_size = dtype.itemsize

    def try_pop(self) -> np.void | None:
        """Non-blocking read. Returns a numpy record or None if empty."""
        head = struct.unpack_from('<Q', self._mm, self._base + HEAD_OFFSET)[0]
        tail = struct.unpack_from('<Q', self._mm, self._base + TAIL_OFFSET)[0]
        if head == tail:
            return None
        slot_offset = self._data_offset + (head % self._capacity) * self._slot_size
        data = np.frombuffer(
            self._mm, dtype=self._dtype, count=1, offset=slot_offset
        )[0].copy()
        struct.pack_into('<Q', self._mm, self._base + HEAD_OFFSET, head + 1)
        return data

    def wait_pop(self, timeout_ms: int = 5000) -> np.void:
        """Spin-wait for an entry with timeout.

        Args:
            timeout_ms: Maximum time to wait in milliseconds.

        Returns:
            A numpy record with the slot data.

        Raises:
            TimeoutError: If no entry arrives within the timeout.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            result = self.try_pop()
            if result is not None:
                return result
        raise TimeoutError(f"Ring buffer read timed out after {timeout_ms}ms")

    def len(self) -> int:
        """Approximate number of items in the ring (best-effort snapshot)."""
        head = struct.unpack_from('<Q', self._mm, self._base + HEAD_OFFSET)[0]
        tail = struct.unpack_from('<Q', self._mm, self._base + TAIL_OFFSET)[0]
        return max(0, tail - head)


class RingWriter:
    """Producer side of an SPSC ring buffer. Writes entries for Zig to read.

    The writer advances the tail counter after writing data into a slot.
    Only one thread/process should write to a given ring instance (the
    Single-Producer guarantee).

    Args:
        mm: An mmap object covering the shared memory region.
        offset: Byte offset within mm where this ring's header starts.
        capacity: Number of slots (must match the consumer's capacity).
        dtype: Numpy dtype describing one slot's layout.
    """

    def __init__(self, mm: mmap.mmap, offset: int, capacity: int, dtype: np.dtype):
        self._mm = mm
        self._base = offset
        self._data_offset = offset + HEADER_SIZE
        self._capacity = capacity
        self._dtype = dtype
        self._slot_size = dtype.itemsize

    def try_push(self, data: np.void) -> bool:
        """Non-blocking write. Returns False if the ring is full.

        Args:
            data: A numpy void (record) matching the ring's dtype.

        Returns:
            True if the item was written, False if the ring is full.
        """
        tail = struct.unpack_from('<Q', self._mm, self._base + TAIL_OFFSET)[0]
        head = struct.unpack_from('<Q', self._mm, self._base + HEAD_OFFSET)[0]
        if tail - head >= self._capacity:
            return False
        slot_offset = self._data_offset + (tail % self._capacity) * self._slot_size
        raw = data.tobytes()
        self._mm[slot_offset:slot_offset + self._slot_size] = raw
        struct.pack_into('<Q', self._mm, self._base + TAIL_OFFSET, tail + 1)
        return True

    def len(self) -> int:
        """Approximate number of items in the ring (best-effort snapshot)."""
        tail = struct.unpack_from('<Q', self._mm, self._base + TAIL_OFFSET)[0]
        head = struct.unpack_from('<Q', self._mm, self._base + HEAD_OFFSET)[0]
        return max(0, tail - head)
