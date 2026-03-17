//! Flint — inference server for large language models.
//!
//! Entry point. Initializes the I/O backend, loads configuration, and
//! dispatches into the server's main loop. In Phase 1 this is a minimal
//! HTTP echo server; later phases add the scheduler, shared-memory IPC,
//! and vLLM worker supervision.

const std = @import("std");
const server = @import("net/server.zig");
const shm_layout = @import("shm/layout.zig");
const ShmRegion = @import("shm/region.zig").ShmRegion;
const BlockTable = @import("block_mgr/block_table.zig").BlockTable;
const Sequence = @import("scheduler/sequence.zig").Sequence;
const Scheduler = @import("scheduler/scheduler.zig").Scheduler;
const SequenceStatus = @import("scheduler/scheduler.zig").SequenceStatus;
const MAX_SEQUENCES = @import("scheduler/scheduler.zig").MAX_SEQUENCES;
const ring_buffer = @import("shm/ring_buffer.zig");
const types = @import("shm/types.zig");
const SpscRing = ring_buffer.SpscRing;

/// Re-exported for integration tests, which import the flint root module
/// and call `handleConnection` directly on a test server.
pub const connection = @import("net/connection.zig");

// Pull in sub-modules so that `zig build test` discovers their test
// blocks transitively from this root.
test {
    _ = @import("http_parser");
    _ = @import("http/response.zig");
    _ = @import("api/router.zig");
    _ = @import("api/openai.zig");
    _ = @import("net/server.zig");
    _ = @import("net/connection.zig");
    _ = @import("shm/region.zig");
    _ = @import("shm/types.zig");
    _ = @import("shm/ring_buffer.zig");
    _ = @import("shm/heartbeat.zig");
    _ = @import("shm/layout.zig");
    _ = @import("block_mgr/allocator.zig");
    _ = @import("block_mgr/block_table.zig");
    _ = @import("scheduler/sequence.zig");
    _ = @import("scheduler/scheduler.zig");
    _ = @import("supervisor/supervisor.zig");
}

/// Number of KV-cache blocks to manage. Determines GPU memory usage for
/// the block allocator. Each block holds TOKENS_PER_BLOCK (16) tokens
/// worth of KV-cache.
const TOTAL_BLOCKS: u32 = 256;

/// Maximum logical blocks per sequence (matches types.MAX_BLOCKS_PER_SEQ).
const MAX_BLOCKS_PER_SEQ: u32 = types.MAX_BLOCKS_PER_SEQ;

/// Ring buffer capacity for schedule/completion rings.
const RING_CAPACITY: u32 = 64;

/// Path for the shared memory file on tmpfs.
const SHM_PATH: [*:0]const u8 = "/dev/shm/flint_server";

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    // Parse optional port argument: `flint [port]`
    // Defaults to 8080 if not provided or invalid.
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next(); // skip program name
    const port: u16 = if (args.next()) |port_str|
        std.fmt.parseInt(u16, port_str, 10) catch 8080
    else
        8080;

    // --- Create shared memory region ---
    const layout = shm_layout.ShmLayout.compute(RING_CAPACITY);
    var region = try ShmRegion.create(SHM_PATH, layout.total_size);
    defer region.close();

    // Zero-initialize the region (ring head/tail must start at 0).
    @memset(region.ptr[0..region.len], 0);

    // --- Allocate scheduler backing buffers ---
    const seq_backing = try gpa.alloc(Sequence, MAX_SEQUENCES);
    defer gpa.free(seq_backing);

    const free_stack = try gpa.alloc(u32, TOTAL_BLOCKS);
    defer gpa.free(free_stack);

    const ref_counts = try gpa.alloc(u16, TOTAL_BLOCKS);
    defer gpa.free(ref_counts);

    // Block table lives in its own heap allocation (not in the shm region
    // for Phase 3 — the mock worker doesn't need it; in Phase 4 it will
    // move to shm).
    const bt_size = BlockTable.totalSize(MAX_SEQUENCES, MAX_BLOCKS_PER_SEQ);
    const block_table_buf = try gpa.alignedAlloc(u8, .@"4", bt_size);
    defer gpa.free(block_table_buf);
    @memset(block_table_buf, 0);

    const seq_status_buf = try gpa.alloc(SequenceStatus, MAX_SEQUENCES);
    defer gpa.free(seq_status_buf);

    const slot_map_buf = try gpa.alloc(u64, MAX_SEQUENCES);
    defer gpa.free(slot_map_buf);

    // --- Initialize scheduler ---
    var scheduler = Scheduler.init(
        seq_backing,
        free_stack,
        ref_counts,
        TOTAL_BLOCKS,
        block_table_buf.ptr,
        0,
        MAX_BLOCKS_PER_SEQ,
        region.ptr,
        layout.schedule_ring_offset,
        RING_CAPACITY,
        region.ptr,
        layout.completion_ring_offset,
        RING_CAPACITY,
        seq_status_buf,
        slot_map_buf,
    );

    // Start the scheduler on a dedicated OS thread.
    try scheduler.start();
    defer scheduler.stop();

    // Run the HTTP server, passing the scheduler for request submission.
    try server.runServer(gpa, io, port, &scheduler);
}
