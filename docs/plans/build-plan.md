# Flint Build Plan

Each phase produces a working, testable deliverable. Phases are additive. The Zig side can be developed and tested without a GPU for phases 1-3 by mocking the Python worker.

---

## Phase 1: TCP Server + HTTP Parser + SSE Echo

**Goal:** Accept TCP connections using `std.Io`, parse HTTP/1.1 via llhttp, send responses. Each connection is a fiber. No proxying — responds to health checks and echoes mock SSE streams.

**Why first:** Everything depends on the network layer. Fully testable without GPU or Python.

### Task 1.1: Project Scaffolding

Set up `build.zig`, `build.zig.zon` (with llhttp dep from pike), directory structure.

### Task 1.2: TCP Accept Loop + Connection Fibers

Files: `src/net/server.zig`, `src/net/connection.zig`

```zig
const std = @import("std");
const Io = std.Io;
const net = Io.net;

pub fn runServer(gpa: std.mem.Allocator, io: Io, port: u16) !void {
    const addr: net.IpAddress = .{ .ip4 = .{ .bytes = .{ 0, 0, 0, 0 }, .port = port } };
    var server = try addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    while (true) {
        const client = server.accept(io) catch |err| {
            log.err("accept failed: {}", .{err});
            continue;
        };
        _ = io.async(handleConnection, .{ gpa, io, client });
    }
}

fn handleConnection(gpa: std.mem.Allocator, io: Io, stream: net.Stream) void {
    defer stream.close(io);

    var recv_buf: [8192]u8 = undefined;
    var send_buf: [8192]u8 = undefined;
    var reader = stream.reader(io, &recv_buf);
    var writer = stream.writer(io, &send_buf);

    while (true) {
        const data = reader.peekGreedy(1) catch break;
        // Parse HTTP, route, respond
        const request = parseAndRoute(data, &writer.interface) catch break;
        reader.toss(data.len);
        if (!request.keep_alive) break;
    }
}
```

**Test:** Open 1000 concurrent TCP connections, verify all accepted.

### Task 1.3: HTTP Parser (llhttp wrapper)

Files: `src/http/parser.zig`, `src/http/types.zig`

Reuse pike's llhttp wrapper pattern:
- `Parser.init(allocator, .request)` — create parser
- `Parser.feed(data)` — feed bytes, fires callbacks
- Callbacks accumulate headers in `ArrayListUnmanaged(u8)` with offset ranges
- On headers_complete, resolve ranges to slices
- `Parser.reset()` — reuse for next request on same connection

Types (adapted from pike):
```zig
pub const Request = struct {
    method: Method,
    url: []const u8,
    headers: []const Header,
    keep_alive: bool,
    content_length: ?u64,
    chunked: bool,
};
```

**Test:** Parse valid requests (GET, POST with body), reject malformed, handle partial reads.

### Task 1.4: SSE Response Writer

File: `src/http/sse.zig`

```zig
pub fn writeHeaders(writer: *Io.Writer) !void {
    try writer.writeAll(
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: keep-alive\r\n\r\n"
    );
    try writer.flush();
}

pub fn writeTokenEvent(writer: *Io.Writer, buf: []u8, token_json: []const u8) !void {
    // Format: "data: {json}\n\n"
    const frame = std.fmt.bufPrint(buf, "data: {s}\n\n", .{token_json}) catch return error.BufferTooSmall;
    try writer.writeAll(frame);
    try writer.flush();
}

pub fn writeDone(writer: *Io.Writer) !void {
    try writer.writeAll("data: [DONE]\n\n");
    try writer.flush();
}
```

### Task 1.5: OpenAI-Compatible API Routing

File: `src/api/openai.zig`

Route parsed requests:
- `GET /health` → `200 OK`
- `GET /v1/models` → JSON model list
- `POST /v1/chat/completions` → parse JSON body, return mock SSE stream (10 fake tokens then `[DONE]`)

For JSON parsing, use `std.json` (available in 0.16).

### Task 1.6: Wire Together — Echo Server

File: `src/main.zig`

```zig
pub fn main(init: std.process.Init) !void {
    try flintMain(init.gpa, init.io);
}

fn flintMain(gpa: std.mem.Allocator, io: std.Io) !void {
    try server.runServer(gpa, io, 8080);
}
```

```bash
curl http://localhost:8080/health
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}'
```

**Deliverable:** Working HTTP server with SSE on io_uring. No GPU, no Python.

**Benchmark target:** 10,000+ concurrent SSE connections.

---

## Phase 2: Shared Memory Primitives + Mock Worker

**Goal:** Build shared memory ring buffers and data types. Test with mock Python worker (no vLLM).

### Task 2.1: Shared Memory Region Manager

File: `src/shm/region.zig`

```zig
pub const ShmRegion = struct {
    ptr: [*]align(4096) u8,
    len: usize,
    fd: i32,

    pub fn create(path: []const u8, size: usize) !ShmRegion { ... }
    pub fn open(path: []const u8) !ShmRegion { ... }
    pub fn close(self: *ShmRegion) void { ... }
    pub fn ptrAt(self: *ShmRegion, comptime T: type, offset: usize) *T { ... }
};
```

Uses `std.posix` for mmap/open/ftruncate. Shared memory doesn't go through `std.Io`.

### Task 2.2: Packed Struct Types (Zig ↔ Python ABI)

File: `src/shm/types.zig`

All `extern struct` with comptime size assertions:
- `RequestT` — sequence ID, token IDs, sampling params
- `ScheduleT` — iteration batch: seq IDs, positions, block tables, swap commands
- `CompletionT` — seq ID, token ID, is_eos, logprob

### Task 2.3: SPSC Ring Buffer

File: `src/shm/ring_buffer.zig`

Generic `SpscRing(comptime T: type)` with cache-line aligned head/tail, atomic load/store with acquire/release ordering.

**Test:** Single-threaded push/pop, multi-threaded producer/consumer with `std.Thread`.

### Task 2.4: Shared Memory Layout

File: `src/shm/layout.zig`

Defines offsets for all regions within a single shm file.

### Task 2.5: Python Shared Memory Bindings

Files: `python/flint_shm/` — mmap + numpy structured arrays mirroring Zig extern structs.

### Task 2.6: Mock Python Worker

File: `python/mock_worker.py` — reads schedules, writes fake completions. No vLLM.

### Task 2.7: Integration Test — Zig ↔ Python Roundtrip

**Deliverable:** Proven shared memory contract between Zig and Python.

---

## Phase 3: Scheduler + Block Allocator

**Goal:** Core scheduling loop and KV-cache block allocator. Scheduler runs on dedicated OS thread (not an Io fiber), posts schedules to shared memory.

### Task 3.1: Sequence State

File: `src/scheduler/sequence.zig`

Flat array of cache-line-aligned `Sequence` structs. 64 bytes each → 1000 seqs = 64KB ≈ L1 cache.

### Task 3.2: Block Allocator

File: `src/block_mgr/allocator.zig`

Free-list stack allocator. O(1) alloc/free. Reference counting for CoW.

**Test:** Alloc until full, free, CoW semantics.

### Task 3.3: Block Table (Shared Memory)

File: `src/block_mgr/block_table.zig`

2D array in shared memory mapping (seq_id, logical_block) → physical_block.

### Task 3.4: Scheduler Core Loop

File: `src/scheduler/scheduler.zig`

Runs on `std.Thread`, not `io.async`. Tight loop:
1. Drain completions from completion ring
2. Update sequence state
3. Compute schedule (promote waiting, continue running, preempt if OOM)
4. Fence block table
5. Push schedule to schedule ring

**Benchmark target:** <5μs per iteration with 256 sequences.

### Task 3.5: Connect Scheduler to HTTP Frontend

Wire Phase 1 HTTP → Phase 3 scheduler:
- POST /v1/chat/completions → create Sequence → push to scheduler
- Connection fibers poll for completions (start with polling, move to eventfd later)
- Token arrives → SSE frame written

**Test:** 100 concurrent requests, all get SSE responses from mock worker.

---

## Phase 4: vLLM Worker Integration

**Goal:** Replace mock worker with real vLLM. Requires GPU.

### Task 4.1: vLLM Worker Loop (`python/flint_worker.py`)

~150 lines. Import vLLM as library, read schedules, call `ModelRunner.execute_model()`, write completions.

### Task 4.2: AttentionMetadata Translation

Translate Zig schedule → vLLM `AttentionMetadata`. Most fragile integration point.

### Task 4.3: Worker Supervisor

File: `src/supervisor/supervisor.zig`

Spawn Python workers via `std.process.Child`, monitor via shm heartbeats, respawn on failure.

### Task 4.4: Tokenizer

Use C FFI to sentencepiece, or have Python worker tokenize initially.

### Task 4.5: End-to-End Test

Test with TinyLlama-1.1B. Real tokens through the full pipeline.

**Deliverable:** Working real inference.

---

## Phase 5: Admission Control + Routing + Production

- Token-aware rate limiter (token bucket draining by output tokens)
- Queue depth backpressure (HTTP 429 fast-reject)
- Prefix cache for routing (rolling hash, worker affinity)
- TOML configuration (`flint.toml`)
- Prometheus /metrics endpoint

---

## Phase 6: NVMe Swap Tier + Weight Loading

- Three-tier KV-cache: GPU → CPU → NVMe
- NVMe I/O through `std.Io` (auto io_uring on Evented)
- `O_DIRECT` for page cache bypass
- Pipelined weight loading: double-buffered NVMe → CPU → GPU

---

## Phase 7 (Future): Multi-Node, LoRA, Disaggregated Prefill/Decode
