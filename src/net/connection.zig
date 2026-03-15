//! Per-connection handler for Flint's HTTP server.
//!
//! Each TCP connection is handled by a single fiber that runs the
//! sequential lifecycle: read request bytes, parse HTTP, route to a
//! handler, write the response. If the client requests keep-alive
//! (HTTP/1.1 default), the fiber loops for the next request on the
//! same connection.
//!
//! ## Why this looks blocking but isn't
//!
//! Every I/O call (`peekGreedy`, `writeAll`, `flush`) goes through the
//! `Io` interface. On `Io.Evented` (Linux production), these calls submit
//! io_uring SQEs and yield the fiber — the runtime resumes it when the
//! CQE arrives. The code reads like synchronous, sequential logic, but
//! thousands of connections run concurrently. On `Io.Threaded` (dev/macOS),
//! the same code runs on a thread pool with real blocking calls.
//!
//! ## Error handling strategy
//!
//! - **Parse errors** (malformed HTTP): write a 400 Bad Request response
//!   and close the connection. We don't attempt recovery because a
//!   desynchronized HTTP stream is unrecoverable.
//! - **Route/handler errors** (internal failures): write a 500 Internal
//!   Server Error and close. The error is logged for debugging.
//! - **Read errors** (except EndOfStream): log and close. The client
//!   likely disconnected or hit a network issue.
//! - **EndOfStream**: the client closed its send side cleanly. We return
//!   without logging — this is normal lifecycle, not an error.
//!
//! Pattern adapted from pike's `src/proxy/connection.zig`, with a key
//! difference: pike forwards HTTP to a backend, while we parse the request
//! and route to our own API handlers.

const std = @import("std");
const net = std.Io.net;
const Io = std.Io;
const log = std.log.scoped(.connection);

const Parser = @import("http_parser").Parser;
const router = @import("../api/router.zig");
const response = @import("../http/response.zig");

/// Handles a single TCP connection through its full lifecycle.
///
/// Reads HTTP requests from `client`, parses them with llhttp, routes to
/// the appropriate handler, and writes responses back. Loops for keep-alive
/// connections. The caller (the accept loop in `server.zig`) spawns this as
/// a concurrent fiber via `io.async`.
pub fn handleConnection(gpa: std.mem.Allocator, io: Io, client: net.Stream) void {
    defer client.close(io);

    // Stack-local buffers for the connection's reader and writer. 8 KiB each
    // is generous for HTTP headers (typical request headers are 1-2 KiB) and
    // well within fiber stack limits.
    var read_buf: [8192]u8 = undefined;
    var write_buf: [8192]u8 = undefined;

    var reader = client.reader(io, &read_buf);
    var writer = client.writer(io, &write_buf);

    // Stack-local HTTP parser. Uses the allocator for header accumulation
    // buffers that may grow across requests (though capacity is retained
    // across keep-alive resets).
    var parser = Parser.init(gpa);
    defer parser.deinit();

    // Keep-alive loop: handle one request per iteration, loop if the client
    // wants to reuse the connection (HTTP/1.1 default).
    while (true) {
        // --- Phase 1: Read and parse the request ---
        //
        // Feed bytes to the parser until it signals message_complete.
        // peekGreedy(1) yields the fiber until at least 1 byte is available,
        // then returns all buffered bytes — zero-copy from the reader's
        // internal buffer.
        while (!parser.message_complete) {
            const data = reader.interface.peekGreedy(1) catch |err| switch (err) {
                error.EndOfStream => return,
                else => {
                    log.debug("read error: {}", .{err});
                    return;
                },
            };

            _ = parser.feed(data) catch |err| {
                log.debug("HTTP parse error: {}", .{err});
                // Malformed request — send 400 and close. No point trying
                // to recover because the stream position is now ambiguous.
                response.writeResponse(
                    &writer.interface,
                    .bad_request,
                    "text/plain",
                    "Bad Request",
                ) catch {};
                return;
            };

            // Consume the bytes from the reader's buffer so the next
            // peekGreedy starts after them.
            reader.interface.toss(data.len);
        }

        // --- Phase 2: Route and respond ---
        const req = parser.request orelse {
            // Parser said message_complete but no request struct — shouldn't
            // happen, but guard against it defensively.
            log.debug("parser completed with no request", .{});
            return;
        };

        log.debug("{s} {s}", .{ @tagName(req.method), req.url });

        router.route(req, parser.body(), &writer.interface) catch |err| {
            log.debug("handler error: {}", .{err});
            // Handler failed — send 500 and close. We don't know what
            // partial data the handler may have written, so it's safest
            // to close the connection.
            response.writeResponse(
                &writer.interface,
                .internal_server_error,
                "text/plain",
                "Internal Server Error",
            ) catch {};
            return;
        };

        // --- Phase 3: Keep-alive or close ---
        if (!req.keep_alive) break;

        // Reset the parser for the next request. This clears parsed state
        // but preserves allocated buffer capacity to avoid re-allocation.
        parser.reset();
    }

    log.debug("connection closed", .{});
}
