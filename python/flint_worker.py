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
    from vllm import LLM, SamplingParams as VllmSamplingParams
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
    """Real vLLM model wrapper using the LLM offline inference API.

    Phase 4 approach: use vLLM's high-level LLM class which manages its
    own KV-cache and scheduling internally. The Flint scheduler submits
    token IDs and sampling params; vLLM handles attention, block management,
    and sampling.

    Future optimization (Phase 5+): bypass LLM/LLMEngine entirely and call
    Worker.execute_model() directly with Zig-managed block tables. This gives
    Flint full control over KV-cache allocation but requires constructing
    SequenceGroupMetadata manually — the most fragile vLLM integration point.

    Requires: GPU + vLLM installed. Pin to vllm==0.6.x.
    """

    def __init__(self, model_name: str, gpu_id: int = 0):
        if not HAS_VLLM:
            raise RuntimeError("vLLM not installed -- cannot use VllmModel")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"Loading model {model_name} on GPU {gpu_id}...", file=sys.stderr)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_num_seqs=256,
            enforce_eager=True,  # Disable CUDA graphs for simpler debugging
            trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"Model loaded. Vocab size: {self.tokenizer.vocab_size}", file=sys.stderr)

    def execute(self, schedule: np.void) -> list[np.void]:
        """Run inference for all sequences in the schedule.

        For each sequence, we generate one token using vLLM's generate().
        This is not maximally efficient (vLLM batches internally, but we
        call generate per-iteration rather than streaming), but it proves
        the pipeline works. Optimization comes in later phases.

        The schedule contains token_ids and sampling params from the Zig
        scheduler. We feed these to vLLM and return CompletionT entries.
        """
        num_seqs = int(schedule['num_sequences'])
        if num_seqs == 0:
            return []

        completions: list[np.void] = []

        # Build prompts from the schedule's token IDs.
        # In Phase 4, the Zig scheduler sends token_ids[i] as the most
        # recent token. For a real implementation we'd need the full
        # token history. For now, use a simple prompt.
        prompts = []
        sampling_params_list = []
        seq_ids = []

        for i in range(num_seqs):
            seq_id = int(schedule['seq_ids'][i])
            seq_ids.append(seq_id)

            # Use the token_id from the schedule as a prompt token.
            # In a real integration, the Zig side would send the full
            # prompt token sequence. For Phase 4, we generate from a
            # minimal prompt.
            token_id = int(schedule['token_ids'][i])
            prompt_token_ids = [token_id] if token_id > 0 else [self.tokenizer.bos_token_id or 1]
            prompts.append({"prompt_token_ids": prompt_token_ids})

            temp = float(schedule['temperatures'][i])
            top_p = float(schedule['top_ps'][i])
            sampling_params_list.append(VllmSamplingParams(
                temperature=max(temp, 0.01),  # vLLM requires temp > 0
                top_p=min(max(top_p, 0.0), 1.0),
                max_tokens=1,  # Generate exactly one token per iteration
            ))

        # Run inference. vLLM batches all sequences internally.
        # We use generate() which handles the full pipeline.
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params_list[0] if len(set(str(s) for s in sampling_params_list)) == 1 else sampling_params_list,
            use_tqdm=False,
        )

        # Convert vLLM outputs to CompletionT entries.
        for idx, output in enumerate(outputs):
            seq_id = seq_ids[idx]
            if output.outputs:
                gen = output.outputs[0]
                token_id = gen.token_ids[-1] if gen.token_ids else 0
                # Check if this is an EOS token
                is_eos = 1 if token_id == (self.tokenizer.eos_token_id or -1) else 0
                logprob = -1.0  # vLLM may not return logprobs by default
                if gen.logprobs and gen.logprobs[-1]:
                    lp = list(gen.logprobs[-1].values())
                    if lp:
                        logprob = lp[0].logprob
            else:
                token_id = 0
                is_eos = 0
                logprob = -1.0

            comp = np.zeros(1, dtype=COMPLETION_DTYPE)
            comp[0]['seq_id'] = seq_id
            comp[0]['token_id'] = token_id
            comp[0]['is_eos'] = is_eos
            comp[0]['logprob'] = np.float16(logprob)
            completions.append(comp[0])

        return completions


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
