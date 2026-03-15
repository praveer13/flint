# CLAUDE.md — Flint

## What is Flint

Flint is an inference server for large language models. Single Zig binary handling all I/O, scheduling, and memory management. GPU computation delegated to vLLM (imported as a Python library, not run as a server). Design principle: CUDA only does matrix multiplication. Everything else belongs in Zig.

Architecture: Zig process communicates with Python GPU workers via shared memory ring buffers. Zig owns networking, scheduling, KV-cache block allocation, and worker lifecycle. Python workers are ~150-line loops calling `vLLM.ModelRunner.execute_model()`.

See `docs/flint-design.md` for full architecture and technical foundations.
See `docs/plans/build-plan.md` for the phased implementation plan.

## Repository Structure

```
flint/
├── CLAUDE.md                    # This file
├── build.zig                    # Zig build system
├── build.zig.zon                # Zig package manifest
├── src/                         # Zig source
│   ├── main.zig                 # Entry point, Io setup, mode dispatch
│   ├── config.zig               # TOML config parser (flint.toml)
│   ├── net/                     # L0 — Network (TCP listener, connections)
│   ├── http/                    # L1 — HTTP parser (llhttp), types, SSE
│   ├── api/                     # OpenAI-compat API (/v1/chat/completions)
│   ├── admission/               # L2 — Rate limiting, backpressure
│   ├── router/                  # L3 — Prefix-aware routing, load balance
│   ├── scheduler/               # L4 — Sequence scheduling, preemption
│   ├── block_mgr/               # L5 — KV-cache block allocator, swap
│   ├── model_mgr/               # L6 — Weight loading, LoRA cache
│   ├── supervisor/              # L7 — Worker lifecycle, health
│   ├── shm/                     # Shared memory primitives (ring buffer, layout)
│   ├── tokenizer/               # BPE tokenizer
│   └── metrics/                 # Prometheus /metrics endpoint
├── python/                      # Python side (vLLM worker + shm bindings)
├── tests/                       # Unit, integration, benchmarks
├── deploy/                      # Dockerfile, flint.toml, fly.toml
└── docs/                        # Design doc and build plans
```

## Build Commands

```bash
zig build                              # Build flint binary
zig build test                         # Run all Zig unit tests
zig build -Doptimize=ReleaseFast       # Optimized release build
zig build run                          # Build and run
```

## Code Philosophy

**Human readability is the top priority.** This codebase should serve as an educational tool for someone new to Zig, systems programming, or vLLM internals. Every file should teach.

- **Explain why, not what** — comments explain reasoning and design decisions, not restate code
- **Module-level docs (`//!`)** — every file starts with a brief explanation of what the module does and how it fits into the architecture
- **Field-level docs (`///`)** — all public struct fields have doc comments
- **Document non-obvious choices** — when code makes an architectural decision (e.g., "scheduler runs on a dedicated OS thread, not a fiber, because..."), explain it inline
- **Name things for what they are** — purpose-specific names over generic ones (`schedule_ring` not `ring`, `block_free_stack` not `stack`)
- **Small, focused modules** — each file does one thing. Split files over ~500 lines. Every directory has a clear domain
- **Flat control flow** — use `orelse return` / early exits instead of nested `if` pyramids
- **Tests as documentation** — test names describe behavior, test bodies show usage patterns
- **Reference pike patterns** — when adapting code from `~/workplace/pike`, add a comment noting the origin so readers can cross-reference

See `~/.claude/zig-best-practices.md` for detailed Zig coding conventions (from Ghostty codebase).

## Zig Conventions (0.16)

- Target Zig 0.16.x
- Entry point: `pub fn main(init: std.process.Init) !void` — use `init.io` and `init.gpa`
- All I/O through `std.Io` interface — functions that do I/O take `io: std.Io` parameter
- Production backend: `Io.Evented` (io_uring on Linux, GCD on macOS) — auto-selected
- TCP: `net.IpAddress` → `.listen(io, .{})` → `.accept(io)` → `stream.reader(io, &buf)` / `stream.writer(io, &buf)`
- Async tasks: `io.async(fn, .{args})` spawns fibers (not `io.concurrent`)
- Reader pattern: `reader.peekGreedy(1)` to peek, `reader.toss(n)` to consume, `writer.writeAll()` + `writer.flush()`
- `extern struct` for all shared-memory types (explicit C ABI layout)
- HTTP parsing via llhttp (C library, same as pike) — not hand-rolled
- Comptime for configuration constants (MAX_BATCH, MAX_BLOCKS, etc.)
- No heap allocation on hot paths
- snake_case for functions/variables, PascalCase for types
- Tests with `zig build test`

## Python Conventions

- Python 3.11+ (vLLM compat)
- Minimal deps: vllm, torch, numpy
- Shared memory via mmap + numpy structured arrays
- No asyncio, no threading — synchronous worker loop
- snake_case everywhere, PascalCase for classes

## Common Pitfalls

1. **Struct layout mismatch** — Zig `extern struct` must exactly match Python `numpy.dtype`. Add comptime size assertions. Test with cross-language roundtrip.
2. **Memory ordering** — Block table updated BEFORE schedule referencing those blocks. Use `@fence(.seq_cst)` between block table writes and schedule ring push.
3. **Ring buffer wraparound** — head/tail are u64, never wrap. Use `index % capacity` for slot offset.
4. **AttentionMetadata** — Most fragile vLLM integration point. Pin vLLM version. Test with real model.
5. **Fiber stack size** — Don't put huge buffers on fiber stack. 64KB recv buf is fine, 1MB is not.
6. **Scheduler is not a fiber** — Runs on dedicated OS thread (`std.Thread`), not `io.async`. It spin-waits on atomics which would block the event loop.
7. **Python GIL** — Worker is single-threaded. No Python threads. Shared-memory design avoids GIL.
8. **Huge pages** — Use `MAP_HUGETLB` for shared memory. Fall back to regular pages if unavailable.
9. **Zig 0.16 API** — Pin to a specific build. APIs may shift between dev builds.
10. **cancel vs await** — Always `defer task.cancel(io) catch {};` after spawning async tasks.

## Key Benchmarks

- Max concurrent SSE connections before degradation
- TTFT: time from HTTP request to first token (p50, p99)
- Token latency between SSE frames (p50, p99)
- Total tokens/second throughput
- Scheduler decision latency (<5μs target for 256 seqs)
- Block alloc/free latency (nanoseconds)
