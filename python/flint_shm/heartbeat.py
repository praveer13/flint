"""Heartbeat region for worker liveness monitoring.

The heartbeat is a single u64 counter in shared memory. The Python worker
increments it after each forward pass. The Zig supervisor reads it
periodically; if the value hasn't changed between two checks, the worker
is considered stuck or dead.

On x86-64, an aligned u64 write is atomic, so no explicit synchronization
is needed beyond the mmap.
"""

import mmap
import struct


class HeartbeatRegion:
    """Read/write access to a u64 heartbeat counter in shared memory.

    Args:
        mm: An mmap object covering the shared memory region.
        offset: Byte offset within mm where the u64 counter lives.
    """

    def __init__(self, mm: mmap.mmap, offset: int):
        self._mm = mm
        self._offset = offset

    def increment(self) -> None:
        """Increment the heartbeat counter by 1."""
        val = struct.unpack_from('<Q', self._mm, self._offset)[0]
        struct.pack_into('<Q', self._mm, self._offset, val + 1)

    def read(self) -> int:
        """Read the current heartbeat counter value."""
        return struct.unpack_from('<Q', self._mm, self._offset)[0]

    def write(self, value: int) -> None:
        """Set the heartbeat counter to a specific value."""
        struct.pack_into('<Q', self._mm, self._offset, value)
