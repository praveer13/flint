//! Flint — inference server for large language models.
//!
//! Entry point. Initializes the I/O backend, loads configuration, and
//! dispatches into the server's main loop. In Phase 1 this is a minimal
//! HTTP echo server; later phases add the scheduler, shared-memory IPC,
//! and vLLM worker supervision.

const std = @import("std");

// Pull in sub-modules so that `zig build test` discovers their test
// blocks transitively from this root.
test {
    _ = @import("http_parser");
    _ = @import("http/response.zig");
    _ = @import("api/router.zig");
    _ = @import("api/openai.zig");
}

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.log.info("flint starting", .{});
}
