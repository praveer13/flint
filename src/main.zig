//! Flint — inference server for large language models.
//!
//! Entry point. Initializes the I/O backend, loads configuration, and
//! dispatches into the server's main loop. In Phase 1 this is a minimal
//! HTTP echo server; later phases add the scheduler, shared-memory IPC,
//! and vLLM worker supervision.

const std = @import("std");
const server = @import("net/server.zig");

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
}

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

    try server.runServer(gpa, io, port);
}
