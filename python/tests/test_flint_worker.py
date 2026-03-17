"""Tests for flint_worker.py running in --mock mode.

Verifies that flint_worker.py (the production worker skeleton) produces
identical behavior to mock_worker.py when run with --mock: reads schedules
from shared memory, writes CompletionT entries with token_id=42, and
exits cleanly on is_shutdown.
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
    """Must match flint_worker.compute_layout and Zig ShmLayout.compute(64)."""
    schedule_ring_size = HEADER_SIZE + SCHEDULE_DTYPE.itemsize * RING_CAPACITY
    schedule_offset = 0
    completion_offset = align_up(schedule_ring_size, 64)
    completion_ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * RING_CAPACITY
    heartbeat_offset = align_up(completion_offset + completion_ring_size, 64)
    raw_total = heartbeat_offset + 8
    total_size = align_up(raw_total, 4096)
    return schedule_offset, completion_offset, heartbeat_offset, total_size


def _spawn_worker(shm_path: str, extra_args: list[str] | None = None):
    """Spawn flint_worker.py --mock as a subprocess."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    worker_script = os.path.join(repo_root, 'flint_worker.py')
    env = {**os.environ, 'PYTHONPATH': repo_root}
    cmd = [sys.executable, worker_script, shm_path, '--mock']
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _wait_ready(heartbeat: HeartbeatRegion, timeout: float = 5.0) -> None:
    """Block until the worker signals ready via heartbeat."""
    deadline = time.monotonic() + timeout
    while heartbeat.read() < 1:
        assert time.monotonic() < deadline, "Worker didn't signal ready in time"
        time.sleep(0.01)


def _collect_completions(
    reader: RingReader, count: int, timeout: float = 5.0,
) -> list[np.void]:
    """Collect *count* completions from the ring, with timeout."""
    results: list[np.void] = []
    deadline = time.monotonic() + timeout
    while len(results) < count:
        assert time.monotonic() < deadline, (
            f"Timed out waiting for completions (got {len(results)}/{count})"
        )
        c = reader.try_pop()
        if c is not None:
            results.append(c)
        else:
            time.sleep(0.01)
    return results


def _send_shutdown(writer: RingWriter) -> None:
    """Push an is_shutdown schedule entry."""
    shutdown = np.zeros(1, dtype=SCHEDULE_DTYPE)
    shutdown[0]['is_shutdown'] = 1
    assert writer.try_push(shutdown[0])


class TestFlintWorkerMock:
    """Tests for flint_worker.py --mock, mirroring the mock_worker tests."""

    def test_single_schedule_three_sequences(self):
        """Send one schedule with 3 sequences, verify 3 completions."""
        sched_off, comp_off, hb_off, total = compute_layout()

        fd_tmp, shm_path = tempfile.mkstemp(prefix='flint_test_worker_')
        try:
            os.ftruncate(fd_tmp, total)
            os.close(fd_tmp)

            fd = os.open(shm_path, os.O_RDWR)
            mm = mmap.mmap(fd, total)
            os.close(fd)

            sched_writer = RingWriter(mm, sched_off, RING_CAPACITY, SCHEDULE_DTYPE)
            comp_reader = RingReader(mm, comp_off, RING_CAPACITY, COMPLETION_DTYPE)
            heartbeat = HeartbeatRegion(mm, hb_off)

            proc = _spawn_worker(shm_path)
            try:
                _wait_ready(heartbeat)

                sched = np.zeros(1, dtype=SCHEDULE_DTYPE)
                sched[0]['iteration_id'] = 1
                sched[0]['num_sequences'] = 3
                sched[0]['seq_ids'][0] = 100
                sched[0]['seq_ids'][1] = 200
                sched[0]['seq_ids'][2] = 300
                sched[0]['is_shutdown'] = 0
                assert sched_writer.try_push(sched[0])

                completions = _collect_completions(comp_reader, 3)

                seq_ids = sorted(int(c['seq_id']) for c in completions)
                assert seq_ids == [100, 200, 300]
                for c in completions:
                    assert int(c['token_id']) == 42
                    assert int(c['is_eos']) == 0
                    assert float(c['logprob']) == pytest.approx(-0.5, abs=0.01)

                assert heartbeat.read() >= 2

                _send_shutdown(sched_writer)
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

        fd_tmp, shm_path = tempfile.mkstemp(prefix='flint_test_worker_')
        try:
            os.ftruncate(fd_tmp, total)
            os.close(fd_tmp)

            fd = os.open(shm_path, os.O_RDWR)
            mm = mmap.mmap(fd, total)
            os.close(fd)

            sched_writer = RingWriter(mm, sched_off, RING_CAPACITY, SCHEDULE_DTYPE)
            comp_reader = RingReader(mm, comp_off, RING_CAPACITY, COMPLETION_DTYPE)
            heartbeat = HeartbeatRegion(mm, hb_off)

            proc = _spawn_worker(shm_path)
            try:
                _wait_ready(heartbeat)

                # Iteration 1: 2 sequences.
                sched1 = np.zeros(1, dtype=SCHEDULE_DTYPE)
                sched1[0]['iteration_id'] = 1
                sched1[0]['num_sequences'] = 2
                sched1[0]['seq_ids'][0] = 10
                sched1[0]['seq_ids'][1] = 20
                assert sched_writer.try_push(sched1[0])

                comps1 = _collect_completions(comp_reader, 2)
                ids1 = sorted(int(c['seq_id']) for c in comps1)
                assert ids1 == [10, 20]

                # Iteration 2: 1 sequence.
                sched2 = np.zeros(1, dtype=SCHEDULE_DTYPE)
                sched2[0]['iteration_id'] = 2
                sched2[0]['num_sequences'] = 1
                sched2[0]['seq_ids'][0] = 999
                assert sched_writer.try_push(sched2[0])

                comps2 = _collect_completions(comp_reader, 1)
                assert int(comps2[0]['seq_id']) == 999
                assert int(comps2[0]['token_id']) == 42

                assert heartbeat.read() >= 3

                _send_shutdown(sched_writer)
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
        """A schedule with num_sequences=0 produces no completions."""
        sched_off, comp_off, hb_off, total = compute_layout()

        fd_tmp, shm_path = tempfile.mkstemp(prefix='flint_test_worker_')
        try:
            os.ftruncate(fd_tmp, total)
            os.close(fd_tmp)

            fd = os.open(shm_path, os.O_RDWR)
            mm = mmap.mmap(fd, total)
            os.close(fd)

            sched_writer = RingWriter(mm, sched_off, RING_CAPACITY, SCHEDULE_DTYPE)
            comp_reader = RingReader(mm, comp_off, RING_CAPACITY, COMPLETION_DTYPE)
            heartbeat = HeartbeatRegion(mm, hb_off)

            proc = _spawn_worker(shm_path)
            try:
                _wait_ready(heartbeat)

                # Send empty schedule.
                empty = np.zeros(1, dtype=SCHEDULE_DTYPE)
                empty[0]['num_sequences'] = 0
                assert sched_writer.try_push(empty[0])

                # Give worker time to process, then verify no completions.
                time.sleep(0.1)
                assert comp_reader.try_pop() is None

                # Confirm worker is still alive with a real schedule.
                sched = np.zeros(1, dtype=SCHEDULE_DTYPE)
                sched[0]['num_sequences'] = 1
                sched[0]['seq_ids'][0] = 777
                assert sched_writer.try_push(sched[0])

                comps = _collect_completions(comp_reader, 1)
                assert int(comps[0]['seq_id']) == 777

                _send_shutdown(sched_writer)
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

    def test_mock_model_unit(self):
        """Unit test MockModel.execute() directly without subprocess."""
        from flint_worker import MockModel

        model = MockModel()

        sched = np.zeros(1, dtype=SCHEDULE_DTYPE)[0]
        sched['num_sequences'] = 2
        sched['seq_ids'][0] = 50
        sched['seq_ids'][1] = 60

        results = model.execute(sched)

        assert len(results) == 2
        ids = sorted(int(r['seq_id']) for r in results)
        assert ids == [50, 60]
        for r in results:
            assert int(r['token_id']) == 42
            assert int(r['is_eos']) == 0
            assert float(r['logprob']) == pytest.approx(-0.5, abs=0.01)

    def test_mock_model_empty_schedule(self):
        """MockModel.execute() returns empty list for zero sequences."""
        from flint_worker import MockModel

        model = MockModel()

        sched = np.zeros(1, dtype=SCHEDULE_DTYPE)[0]
        sched['num_sequences'] = 0

        results = model.execute(sched)
        assert results == []
