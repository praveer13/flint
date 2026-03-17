#!/usr/bin/env python3
"""Mock GPU worker for testing the shm pipeline without a real GPU.

Usage: python mock_worker.py <shm_path>

Reads ScheduleT entries from the schedule ring, writes fake CompletionT
entries (token_id=42) for each sequence, and increments the heartbeat
counter. Exits when a schedule with is_shutdown=1 is received.
"""

import sys
import mmap
import os

# Ensure the python/ directory is on sys.path so `flint_shm` is importable
# regardless of the working directory or how the script is invoked.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np

from flint_shm.types import COMPLETION_DTYPE, SCHEDULE_DTYPE
from flint_shm.ring_buffer import RingReader, RingWriter, HEADER_SIZE
from flint_shm.heartbeat import HeartbeatRegion

RING_CAPACITY = 64


def align_up(value: int, alignment: int) -> int:
    """Round *value* up to the next multiple of *alignment* (power of two)."""
    return (value + alignment - 1) & ~(alignment - 1)


def compute_layout(ring_capacity: int = RING_CAPACITY):
    """Return (schedule_offset, completion_offset, heartbeat_offset, total_size).

    Must match ``ShmLayout.compute()`` in ``src/shm/layout.zig``.
    """
    schedule_ring_size = HEADER_SIZE + SCHEDULE_DTYPE.itemsize * ring_capacity
    schedule_offset = 0

    completion_offset = align_up(schedule_ring_size, 64)
    completion_ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * ring_capacity

    heartbeat_offset = align_up(completion_offset + completion_ring_size, 64)

    # Zig's HeartbeatRegion is a single u64 (8 bytes).
    raw_total = heartbeat_offset + 8
    total_size = align_up(raw_total, 4096)

    return schedule_offset, completion_offset, heartbeat_offset, total_size


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <shm_path> [--gpu <id>]", file=sys.stderr)
        sys.exit(1)

    shm_path = sys.argv[1]
    # Extra arguments (e.g., --gpu 0) are accepted but ignored by the mock.

    # Open shared memory file.
    fd = os.open(shm_path, os.O_RDWR)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size)
    os.close(fd)

    sched_off, comp_off, hb_off, _total = compute_layout()

    # Create ring buffer views.
    schedules = RingReader(mm, sched_off, RING_CAPACITY, SCHEDULE_DTYPE)
    completions = RingWriter(mm, comp_off, RING_CAPACITY, COMPLETION_DTYPE)
    heartbeat = HeartbeatRegion(mm, hb_off)

    # Signal ready by setting heartbeat to 1.
    heartbeat.write(1)

    # Main loop: read schedules, write fake completions.
    while True:
        try:
            sched = schedules.wait_pop(timeout_ms=10000)
        except TimeoutError:
            # No schedule received within the timeout. This happens when the
            # server has no active requests or when it has been terminated.
            # Re-check and continue — the supervisor will detect us as stalled
            # if we stop incrementing the heartbeat.
            continue

        if int(sched['is_shutdown']):
            break

        num_seqs = int(sched['num_sequences'])
        for i in range(num_seqs):
            comp = np.zeros(1, dtype=COMPLETION_DTYPE)
            comp[0]['seq_id'] = sched['seq_ids'][i]
            comp[0]['token_id'] = 42
            comp[0]['is_eos'] = 0
            comp[0]['logprob'] = np.float16(-0.5)
            completions.try_push(comp[0])

        heartbeat.increment()

    mm.close()


if __name__ == '__main__':
    main()
