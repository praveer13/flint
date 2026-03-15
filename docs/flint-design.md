# Flint Design Document

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Flint (single Zig binary)                     │
│                                                                      │
│  L0 ─ Net          std.Io (io_uring on Linux), TLS, connections     │
│  L1 ─ Protocol     HTTP/1.1 (llhttp), SSE streaming, OpenAI API    │
│  L2 ─ Admission    Rate limiting, auth, quota, request validation    │
│  L3 ─ Router       Prefix-aware routing, LoRA affinity, load balance│
│  L4 ─ Scheduler    Sequence scheduling, preemption, fairness        │
│  L5 ─ BlockMgr     KV-cache page allocator, swap engine, CoW       │
│  L6 ─ ModelMgr     Weight loading, LoRA hot-swap, quantization      │
│  L7 ─ Supervisor   Worker lifecycle, health, metrics                │
│                                                                      │
├──────────────────── shared memory boundary ──────────────────────────┤
│                                                                      │
│  vLLM Worker (Python)                                                │
│  - vLLM imported as library, not running as server                   │
│  - Reads schedule from shm ring buffer                               │
│  - Calls vLLM ModelRunner.execute_model() for forward pass           │
│  - Uses vLLM's kernels: flash attn, paged KV, quantized matmul      │
│  - Writes output tokens to shm                                      │
│  - No scheduler. No networking. No block allocation. No disk I/O.   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Technical Foundations

### Zig 0.16 and std.Io

Flint targets Zig 0.16.x. The defining feature is `std.Io` — a new interface for all I/O operations. Functions that do I/O accept an `io: std.Io` parameter.

Zig 0.16 provides several `Io` backends:

- **`Io.Blocking`** — maps directly to blocking syscalls
- **`Io.Threaded`** — blocking I/O multiplexed over a thread pool
- **`Io.Evented`** — production backend. Uses io_uring on Linux, GCD on macOS. Green threads (fibers) on top of event loop. Code looks sequential but runs async.

**Entry point (0.16):**
```zig
pub fn main(init: std.process.Init) !void {
    const io = init.io;   // Auto-selected backend
    const gpa = init.gpa; // General purpose allocator
    try flintMain(gpa, io);
}
```

**Network I/O pattern (from pike — proven working):**
```zig
const net = std.Io.net;

// Listen
const addr: net.IpAddress = .{ .ip4 = .{ .bytes = .{ 0, 0, 0, 0 }, .port = 8080 } };
var server = try addr.listen(io, .{ .reuse_address = true });
defer server.deinit(io);

// Accept loop — each connection becomes a fiber
while (true) {
    const client = server.accept(io) catch |err| {
        log.err("accept failed: {}", .{err});
        continue;
    };
    _ = io.async(handleConnection, .{ io, client });
}

// Connection handler — sequential code, async execution
fn handleConnection(io: Io, stream: net.Stream) void {
    defer stream.close(io);
    var buf: [8192]u8 = undefined;
    var reader = stream.reader(io, &buf);
    var wbuf: [8192]u8 = undefined;
    var writer = stream.writer(io, &wbuf);

    // peekGreedy blocks fiber until data available
    const data = reader.peekGreedy(1) catch return;
    // ... parse, respond ...
    writer.writeAll(response) catch return;
    writer.flush() catch return;
    reader.toss(data.len);
}
```

**What std.Io gives us for free:**
- io_uring submission batching and completion reaping
- Fiber scheduling across OS threads
- Cancellation (cancel a fiber = cancel all its pending I/O)
- File I/O in same event loop (NVMe reads for weight loading)

**What we may need raw access for (optimize later):**
- Registered buffers (`IORING_REGISTER_BUFFERS`) for zero-copy recv
- `IORING_ACCEPT_MULTISHOT` for batch accepting
- `O_DIRECT` flags for NVMe bypass
- kTLS handoff after TLS handshake

**Strategy: Start with std.Io everywhere. Drop to raw io_uring only where profiling demands it.**

### HTTP Parsing via llhttp

We use llhttp (from Node.js) via C FFI, same as pike. This gives us a battle-tested HTTP/1.1 parser without reinventing it. The parser is wrapped in a Zig-idiomatic API with zero-copy header accumulation.

Key pattern from pike:
- C callbacks with `callconv(.c)` fire as llhttp scans the buffer
- Header fragments accumulated in `ArrayListUnmanaged(u8)` with offset/length ranges
- Ranges resolved to slices once headers complete
- Parser is feed-based: call `parser.feed(data)` with whatever bytes are available
- Combined with `reader.peekGreedy(1)` + `reader.toss(n)` for incremental parsing

### Shared Memory IPC

Zig process and Python workers communicate through mmap'd files on `/dev/shm/`. Both sides map the same file and read/write directly — no serialization, no sockets, no GIL involvement.

Primary primitive: SPSC (Single-Producer, Single-Consumer) ring buffer:
- Atomic `head` (consumer) and `tail` (producer), cache-line separated
- Fixed-size slots with packed structs
- No locks — correctness from atomic ordering and single-writer discipline
- Producer: write data, then `@atomicStore(.release)` on tail
- Consumer: `@atomicLoad(.acquire)` on tail, read data, advance head

### KV-Cache and PagedAttention

During inference, the model maintains a Key-Value cache. For a 70B model, each token's KV-cache is ~1.2MB. A 4096-token sequence uses ~5GB.

PagedAttention (from vLLM) manages this like OS virtual memory:
- KV-cache divided into fixed-size blocks (e.g., 16 tokens per block)
- Block table maps (sequence_id, logical_block) → physical GPU address
- Blocks allocated on demand
- Shared prefixes use same physical blocks (Copy-on-Write)
- Blocks swappable to CPU or disk when GPU memory full

Flint's block manager owns all metadata (block table, free list, ref counts) in Zig. Actual GPU memory allocated by vLLM worker. Block table in shared memory for worker to read during attention.

### vLLM as a Library

We import only the compute layer:
- `vllm.model_executor.model_loader.get_model()` — loads model architectures
- `ModelRunner.execute_model()` — one forward pass
- `Sampler` — next-token sampling (top-p, top-k, temperature)
- `AttentionMetadata` — tells attention kernel about sequences and KV-cache blocks

We do NOT use vLLM's server, scheduler, block manager, API layer, or async engine.

Pin to vLLM `v0.6.x`. All vLLM-touching code lives in `python/`.
