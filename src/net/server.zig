//! TCP accept loop for Flint's network layer.
//!
//! Listens on a configurable port with SO_REUSEADDR enabled and spawns a
//! connection handler fiber for each accepted client. Handlers run
//! concurrently via `io.async` — fire-and-forget at this layer.
//!
//! The accept loop never exits on transient errors (fd exhaustion, etc.).
//! It logs the error and keeps accepting. Only fatal listen errors
//! propagate to the caller.
//!
//! ## Fiber-per-connection model
//!
//! Each accepted connection becomes a green thread (fiber) managed by the
//! `Io` backend. On `Io.Evented` (Linux production), fibers are scheduled
//! on io_uring — the accept call submits an `IORING_OP_ACCEPT` SQE and
//! yields, so thousands of connections can be served concurrently without
//! OS threads. On `Io.Threaded` (development/macOS), fibers map to a
//! thread pool. The accept loop and connection handlers use the same
//! sequential code in both cases.
//!
//! Pattern adapted from pike's `src/proxy/listener.zig`.

const std = @import("std");
const net = std.Io.net;
const Io = std.Io;
const connection = @import("connection.zig");
const Scheduler = @import("../scheduler/scheduler.zig").Scheduler;
const log = std.log.scoped(.server);

/// Runs the TCP accept loop: binds to 0.0.0.0:`port`, accepts connections,
/// and spawns a `handleConnection` fiber for each one.
///
/// This function blocks forever (it is the server's main loop). It only
/// returns if the initial `listen()` call fails — accept-time errors are
/// logged and swallowed so that one bad connection never brings down the
/// server.
///
/// `scheduler` is optional: when non-null, requests to `/v1/chat/completions`
/// are routed through the scheduler and tokens stream from the GPU worker
/// via shared memory. When null, the handler falls back to mock tokens
/// (Phase 1 behavior), which is useful for integration tests that don't
/// need a scheduler.
pub fn runServer(gpa: std.mem.Allocator, io: Io, port: u16, scheduler: ?*Scheduler) !void {
    const addr: net.IpAddress = .{ .ip4 = .{ .bytes = .{ 0, 0, 0, 0 }, .port = port } };
    var server = try addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    log.info("listening on port {d}", .{port});

    while (true) {
        const client = server.accept(io) catch |err| {
            log.err("accept failed: {}", .{err});
            continue;
        };

        // Fire-and-forget: spawn connection handler as an async fiber.
        // Each fiber runs the sequential read-parse-route-respond loop
        // in connection.zig. We don't track the handle here — graceful
        // shutdown (draining in-flight connections) is a later concern.
        _ = io.async(connection.handleConnection, .{ gpa, io, client, scheduler });
    }
}
