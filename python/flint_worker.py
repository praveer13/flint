#!/usr/bin/env python3
"""Flint GPU worker -- runs inference using vLLM as a library.

This is the ~150-line Python process that does the actual GPU work.
It reads schedules from shared memory, calls vLLM's ModelRunner for
forward passes, samples output tokens, and writes completions back.

The Zig server handles everything else: networking, scheduling,
block management, process supervision.

Usage:
    python flint_worker.py <shm_path> [--model <model_name>] [--gpu <gpu_id>]
    python flint_worker.py <shm_path> --mock
"""

import argparse
import mmap
import os
import sys

import numpy as np

from flint_shm.types import COMPLETION_DTYPE, SCHEDULE_DTYPE
from flint_shm.ring_buffer import RingReader, RingWriter, HEADER_SIZE
from flint_shm.heartbeat import HeartbeatRegion

# vLLM imports -- gated so the worker can run in mock mode without GPU.
try:
    import torch
    from vllm.model_executor.model_loader import get_model  # noqa: F401
    from vllm.worker.model_runner import ModelRunner  # noqa: F401

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


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

    raw_total = heartbeat_offset + 8
    total_size = align_up(raw_total, 4096)

    return schedule_offset, completion_offset, heartbeat_offset, total_size


class MockModel:
    """Mock model for testing without GPU/vLLM.

    Returns token_id=42 for every sequence, simulating a model that
    always predicts the same token. Used for pipeline testing.
    """

    def execute(self, schedule: np.void) -> list[np.void]:
        """Return one fake completion per sequence."""
        completions: list[np.void] = []
        num_seqs = int(schedule['num_sequences'])
        for i in range(num_seqs):
            comp = np.zeros(1, dtype=COMPLETION_DTYPE)
            comp[0]['seq_id'] = schedule['seq_ids'][i]
            comp[0]['token_id'] = 42
            comp[0]['is_eos'] = 0
            comp[0]['logprob'] = np.float16(-0.5)
            completions.append(comp[0])
        return completions


class VllmModel:
    """Real vLLM model wrapper.

    Loads a model using vLLM's model loader and runs forward passes
    using ModelRunner.execute_model(). This is the production path.

    NOTE: This requires a GPU and vLLM installed. Not functional in
    Phase 4 skeleton -- will be completed when GPU testing is available.
    """

    def __init__(self, model_name: str, gpu_id: int = 0):
        if not HAS_VLLM:
            raise RuntimeError("vLLM not installed -- cannot use VllmModel")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # TODO(phase4): Load model via vLLM's model loader.
        # self.device = torch.device(f"cuda:0")
        # self.model_config = ...
        # self.model_runner = ModelRunner(...)
        # self.model_runner.load_model()
        raise NotImplementedError(
            "VllmModel not yet implemented -- needs GPU testing"
        )

    def execute(self, schedule: np.void) -> list[np.void]:
        """Run a forward pass and sample output tokens.

        Translates the ScheduleT into vLLM's AttentionMetadata,
        calls execute_model() for the forward pass, samples from
        the logits, and returns CompletionT entries.
        """
        # TODO(phase4): Build AttentionMetadata from schedule fields:
        #   - seq_ids, seq_lens, positions for sequence metadata
        #   - block_tables, num_blocks for paged attention
        #   - is_prefill to distinguish prefill vs decode
        # TODO(phase4): Call self.model_runner.execute_model(...)
        # TODO(phase4): Sample tokens from logits using temperatures/top_ps
        # TODO(phase4): Return CompletionT entries
        raise NotImplementedError()


def main() -> None:
    parser = argparse.ArgumentParser(description="Flint GPU worker")
    parser.add_argument("shm_path", help="Path to shared memory file")
    parser.add_argument("--model", default=None, help="Model name or path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model (no GPU required)",
    )
    args = parser.parse_args()

    # Open shared memory.
    fd = os.open(args.shm_path, os.O_RDWR)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size)
    os.close(fd)

    sched_off, comp_off, hb_off, _total = compute_layout()

    # Create ring buffer views.
    schedules = RingReader(mm, sched_off, RING_CAPACITY, SCHEDULE_DTYPE)
    completions = RingWriter(mm, comp_off, RING_CAPACITY, COMPLETION_DTYPE)
    heartbeat = HeartbeatRegion(mm, hb_off)

    # Load model.
    if args.mock or not HAS_VLLM:
        model = MockModel()
    else:
        if args.model is None:
            print("Error: --model is required when not using --mock", file=sys.stderr)
            sys.exit(1)
        model = VllmModel(args.model, args.gpu)

    # Signal ready.
    heartbeat.write(1)

    # Main loop: read schedule -> execute model -> write completions.
    while True:
        sched = schedules.wait_pop(timeout_ms=10000)

        if int(sched['is_shutdown']):
            break

        results = model.execute(sched)

        for comp in results:
            completions.try_push(comp)

        heartbeat.increment()

    mm.close()


if __name__ == '__main__':
    main()
