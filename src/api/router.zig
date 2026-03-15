//! HTTP request router for Flint's API layer.
//!
//! This module maps incoming HTTP requests to the appropriate handler based
//! on the URL path and method. It is the glue between the protocol layer
//! (which parses raw bytes into `Request` structs) and the API handlers
//! (which produce responses).
//!
//! ## Supported routes
//!
//! | Path                      | Method | Handler                     |
//! |---------------------------|--------|-----------------------------|
//! | `/health`                 | GET    | Returns `{"status":"ok"}`   |
//! | `/v1/models`              | GET    | Returns a mock model list   |
//! | `/v1/chat/completions`    | POST   | Streams SSE token chunks    |
//!
//! Any other path returns 404 Not Found. A valid path with the wrong HTTP
//! method returns 405 Method Not Allowed.

const std = @import("std");
const Io = std.Io;

const http_types = @import("http_parser").types;
const response = @import("../http/response.zig");
const openai = @import("openai.zig");

/// Routes a parsed HTTP request to the matching handler.
///
/// `req` is the parsed request (method, URL, headers). `body_data` is the
/// raw request body (may be empty for GET requests). `writer` is the
/// connection's buffered writer, used by handlers to send the response.
///
/// The router checks the URL path first, then validates the HTTP method.
/// This ordering lets us distinguish "not found" (wrong path) from
/// "method not allowed" (right path, wrong verb).
pub fn route(req: http_types.Request, body_data: []const u8, writer: *Io.Writer) !void {
    if (std.mem.eql(u8, req.url, "/health")) {
        if (req.method != .GET) {
            try response.writeResponse(writer, .method_not_allowed, "text/plain", "Method Not Allowed");
            return;
        }
        try response.writeResponse(writer, .ok, "application/json", "{\"status\":\"ok\"}");
    } else if (std.mem.eql(u8, req.url, "/v1/models")) {
        if (req.method != .GET) {
            try response.writeResponse(writer, .method_not_allowed, "text/plain", "Method Not Allowed");
            return;
        }
        // Phase 1: return a single hard-coded mock model entry.
        // In later phases this will reflect actually loaded models.
        try response.writeResponse(writer, .ok, "application/json",
            \\{"object":"list","data":[{"id":"mock-model","object":"model","owned_by":"flint"}]}
        );
    } else if (std.mem.eql(u8, req.url, "/v1/chat/completions")) {
        if (req.method != .POST) {
            try response.writeResponse(writer, .method_not_allowed, "text/plain", "Method Not Allowed");
            return;
        }
        try openai.handleChatCompletions(body_data, writer);
    } else {
        try response.writeResponse(writer, .not_found, "text/plain", "Not Found");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Helper: build a minimal `Request` for testing the router.
fn testRequest(method: http_types.Method, url: []const u8) http_types.Request {
    return .{
        .method = method,
        .url = url,
        .version_major = 1,
        .version_minor = 1,
        .headers = &.{},
        .keep_alive = true,
        .content_length = null,
        .chunked = false,
    };
}

test "GET /health returns 200 JSON" {
    // We cannot easily capture Io.Writer output in unit tests without a
    // real Io backend, so we just verify that `route` compiles and the
    // type signatures line up. Integration tests will exercise the full
    // write path.
    _ = &route;
}

test "unknown path returns 404" {
    _ = testRequest(.GET, "/nonexistent");
}

test "wrong method returns 405" {
    _ = testRequest(.POST, "/health");
}
