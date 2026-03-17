"""Cross-language shared memory roundtrip test.

Simulates the Zig scheduler -> Python worker -> Zig scheduler data flow
entirely in Python, proving the shared memory contract works end-to-end.

The test creates a temporary file (simulating /dev/shm/), writes schedule
entries from the "Zig side" (this test), spawns mock_worker.py as a
subprocess to play the GPU worker role, and reads back completion entries
to verify correctness.
"""

import mmap
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

from flint_shm.types import COMPLETION_DTYPE, SCHEDULE_DTYPE
from flint_shm.ring_buffer import HEADER_SIZE, RingReader, RingWriter
from flint_shm.heartbeat import HeartbeatRegion

RING_CAPACITY = 64


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def compute_layout():
    """Must match mock_worker.compute_layout and Zig ShmLayout.compute(64)."""
    schedule_ring_size = HEADER_SIZE + SCHEDULE_DTYPE.itemsize * RING_CAPACITY
    schedule_offset = 0
    completion_offset = align_up(schedule_ring_size, 64)
    completion_ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * RING_CAPACITY
    heartbeat_offset = align_up(completion_offset + completion_ring_size, 64)
    raw_total = heartbeat_offset + 8
    total_size = align_up(raw_total, 4096)
    return schedule_offset, completion_offset, heartbeat_offset, total_size


class TestShmRoundtrip:
    """End-to-end shared memory roundtrip: writer -> mock_worker -> reader."""

    def test_single_schedule_three_sequences(self):
        """Send one schedule with 3 sequences, verify 3 completions come back."""
        sched_off, comp_off, hb_off, total = compute_layout()

        # Create a temporary file to act as shared memory.
        fd_tmp, shm_path = tempfile.mkstemp(prefix='flint_test_shm_')
        try:
            os.ftruncate(fd_tmp, total)
            os.close(fd_tmp)

            # mmap the file.
            fd = os.open(shm_path, os.O_RDWR)
            mm = mmap.mmap(fd, total)
            os.close(fd)

            # "Zig side" views: we write schedules, read completions.
            sched_writer = RingWriter(mm, sched_off, RING_CAPACITY, SCHEDULE_DTYPE)
            comp_reader = RingReader(mm, comp_off, RING_CAPACITY, COMPLETION_DTYPE)
            heartbeat = HeartbeatRegion(mm, hb_off)

            # Spawn mock worker.
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            worker_script = os.path.join(repo_root, 'mock_worker.py')
            env = {**os.environ, 'PYTHONPATH': repo_root}
            proc = subprocess.Popen(
                [sys.executable, worker_script, shm_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                # Wait for the worker to signal ready (heartbeat == 1).
                deadline = time.monotonic() + 5.0
                while heartbeat.read() < 1:
                    assert time.monotonic() < deadline, "Worker didn't signal ready within 5s"
                    time.sleep(0.01)

                # Write a schedule with 3 sequences.
                sched = np.zeros(1, dtype=SCHEDULE_DTYPE)
                sched[0]['iteration_id'] = 1
                sched[0]['num_sequences'] = 3
                sched[0]['seq_ids'][0] = 100
                sched[0]['seq_ids'][1] = 200
                sched[0]['seq_ids'][2] = 300
                sched[0]['is_shutdown'] = 0
                assert sched_writer.try_push(sched[0]), "Schedule ring unexpectedly full"

                # Read 3 completions back.
                completions = []
                deadline = time.monotonic() + 5.0
                while len(completions) < 3:
                    assert time.monotonic() < deadline, (
                        f"Timed out waiting for completions (got {len(completions)}/3)"
                    )
                    c = comp_reader.try_pop()
                    if c is not None:
                        completions.append(c)
                    else:
                        time.sleep(0.01)

                # Verify completions.
                seq_ids = sorted(int(c['seq_id']) for c in completions)
                assert seq_ids == [100, 200, 300]
                for c in completions:
                    assert int(c['token_id']) == 42
                    assert int(c['is_eos']) == 0
                    assert float(c['logprob']) == pytest.approx(-0.5, abs=0.01)

                # Heartbeat should have been incremented once (ready=1, then +1).
                assert heartbeat.read() >= 2

                # Send shutdown.
                shutdown = np.zeros(1, dtype=SCHEDULE_DTYPE)
                shutdown[0]['is_shutdown'] = 1
                assert sched_writer.try_push(shutdown[0])

                proc.wait(timeout=5)
                assert proc.returncode == 0

            except Exception:
                proc.kill()
                proc.wait(timeout=2)
                raise
            finally:
                mm.close()
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_multiple_iterations(self):
        """Send two schedules in sequence, verify completions for both."""
        sched_off, comp_off, hb_off, total = compute_layout()

        fd_tmp, shm_path = tempfile.mkstemp(prefix='flint_test_shm_')
        try:
            os.ftruncate(fd_tmp, total)
            os.close(fd_tmp)

            fd = os.open(shm_path, os.O_RDWR)
            mm = mmap.mmap(fd, total)
            os.close(fd)

            sched_writer = RingWriter(mm, sched_off, RING_CAPACITY, SCHEDULE_DTYPE)
            comp_reader = RingReader(mm, comp_off, RING_CAPACITY, COMPLETION_DTYPE)
            heartbeat = HeartbeatRegion(mm, hb_off)

            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            worker_script = os.path.join(repo_root, 'mock_worker.py')
            env = {**os.environ, 'PYTHONPATH': repo_root}
            proc = subprocess.Popen(
                [sys.executable, worker_script, shm_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                # Wait for ready.
                deadline = time.monotonic() + 5.0
                while heartbeat.read() < 1:
                    assert time.monotonic() < deadline, "Worker didn't signal ready"
                    time.sleep(0.01)

                # --- Iteration 1: 2 sequences ---
                sched1 = np.zeros(1, dtype=SCHEDULE_DTYPE)
                sched1[0]['iteration_id'] = 1
                sched1[0]['num_sequences'] = 2
                sched1[0]['seq_ids'][0] = 10
                sched1[0]['seq_ids'][1] = 20
                assert sched_writer.try_push(sched1[0])

                completions1 = []
                deadline = time.monotonic() + 5.0
                while len(completions1) < 2:
                    assert time.monotonic() < deadline, "Timed out on iteration 1"
                    c = comp_reader.try_pop()
                    if c is not None:
                        completions1.append(c)
                    else:
                        time.sleep(0.01)

                ids1 = sorted(int(c['seq_id']) for c in completions1)
                assert ids1 == [10, 20]

                # --- Iteration 2: 1 sequence ---
                sched2 = np.zeros(1, dtype=SCHEDULE_DTYPE)
                sched2[0]['iteration_id'] = 2
                sched2[0]['num_sequences'] = 1
                sched2[0]['seq_ids'][0] = 999
                assert sched_writer.try_push(sched2[0])

                deadline = time.monotonic() + 5.0
                c2 = None
                while c2 is None:
                    assert time.monotonic() < deadline, "Timed out on iteration 2"
                    c2 = comp_reader.try_pop()
                    if c2 is None:
                        time.sleep(0.01)

                assert int(c2['seq_id']) == 999
                assert int(c2['token_id']) == 42

                # Heartbeat: ready(1) + 2 iterations = 3
                assert heartbeat.read() >= 3

                # Shutdown.
                shutdown = np.zeros(1, dtype=SCHEDULE_DTYPE)
                shutdown[0]['is_shutdown'] = 1
                assert sched_writer.try_push(shutdown[0])

                proc.wait(timeout=5)
                assert proc.returncode == 0

            except Exception:
                proc.kill()
                proc.wait(timeout=2)
                raise
            finally:
                mm.close()
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_zero_sequence_schedule(self):
        """A schedule with num_sequences=0 should produce no completions."""
        sched_off, comp_off, hb_off, total = compute_layout()

        fd_tmp, shm_path = tempfile.mkstemp(prefix='flint_test_shm_')
        try:
            os.ftruncate(fd_tmp, total)
            os.close(fd_tmp)

            fd = os.open(shm_path, os.O_RDWR)
            mm = mmap.mmap(fd, total)
            os.close(fd)

            sched_writer = RingWriter(mm, sched_off, RING_CAPACITY, SCHEDULE_DTYPE)
            comp_reader = RingReader(mm, comp_off, RING_CAPACITY, COMPLETION_DTYPE)
            heartbeat = HeartbeatRegion(mm, hb_off)

            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            worker_script = os.path.join(repo_root, 'mock_worker.py')
            env = {**os.environ, 'PYTHONPATH': repo_root}
            proc = subprocess.Popen(
                [sys.executable, worker_script, shm_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                deadline = time.monotonic() + 5.0
                while heartbeat.read() < 1:
                    assert time.monotonic() < deadline, "Worker didn't signal ready"
                    time.sleep(0.01)

                # Send empty schedule.
                empty = np.zeros(1, dtype=SCHEDULE_DTYPE)
                empty[0]['num_sequences'] = 0
                assert sched_writer.try_push(empty[0])

                # Give worker time to process, then check no completions appeared.
                # We send a second schedule (with 1 seq) to confirm the worker
                # is still alive and processed the empty one.
                time.sleep(0.1)
                assert comp_reader.try_pop() is None, "Got unexpected completion for empty schedule"

                # Confirm the worker is alive by sending a real schedule.
                sched = np.zeros(1, dtype=SCHEDULE_DTYPE)
                sched[0]['num_sequences'] = 1
                sched[0]['seq_ids'][0] = 777
                assert sched_writer.try_push(sched[0])

                deadline = time.monotonic() + 5.0
                c = None
                while c is None:
                    assert time.monotonic() < deadline, "Worker seems stuck"
                    c = comp_reader.try_pop()
                    if c is None:
                        time.sleep(0.01)
                assert int(c['seq_id']) == 777

                # Shutdown.
                shutdown = np.zeros(1, dtype=SCHEDULE_DTYPE)
                shutdown[0]['is_shutdown'] = 1
                assert sched_writer.try_push(shutdown[0])

                proc.wait(timeout=5)
                assert proc.returncode == 0

            except Exception:
                proc.kill()
                proc.wait(timeout=2)
                raise
            finally:
                mm.close()
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)
