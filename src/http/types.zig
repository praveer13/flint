//! HTTP request and response types for Flint's protocol layer.
//!
//! These are pure data types with no I/O dependency — they can be used by both
//! the parser (`parser.zig`) and the response writer without pulling in
//! networking code. The parser populates a `Request` with slices pointing into
//! the connection's recv buffer (zero-copy), and downstream handlers read those
//! slices to route and respond.
//!
//! The `Method` enum bridges llhttp's integer method codes to a Zig-native
//! enum so that the rest of the codebase never deals with raw integers.

const std = @import("std");

/// Standard HTTP methods that Flint recognises.
///
/// The enum values are *not* tied to llhttp codes — use `fromLlhttp` to
/// convert. We only include methods that make sense for an inference API
/// server; exotic methods (WebDAV, RTSP, etc.) are intentionally absent.
pub const Method = enum {
    DELETE,
    GET,
    HEAD,
    POST,
    PUT,
    CONNECT,
    OPTIONS,
    TRACE,
    PATCH,

    /// Map an llhttp integer method code to a `Method`.
    ///
    /// llhttp assigns fixed integers to each HTTP method (e.g. GET = 1,
    /// POST = 3). This function translates those codes into our enum.
    /// Returns `null` for any code we don't handle — callers should treat
    /// that as an unsupported-method error.
    pub fn fromLlhttp(code: u8) ?Method {
        return switch (code) {
            0 => .DELETE,
            1 => .GET,
            2 => .HEAD,
            3 => .POST,
            4 => .PUT,
            5 => .CONNECT,
            6 => .OPTIONS,
            7 => .TRACE,
            28 => .PATCH,
            else => null,
        };
    }
};

/// A single HTTP header as a name/value pair.
///
/// Both slices point into the connection's recv buffer — no copies are made
/// during parsing. The slices are only valid for the lifetime of that buffer.
pub const Header = struct {
    /// Header field name, e.g. `"Content-Type"`. Not normalised to any
    /// particular case — use `std.ascii.eqlIgnoreCase` for comparisons.
    name: []const u8,

    /// Header field value, e.g. `"application/json"`.
    value: []const u8,
};

/// A parsed HTTP/1.1 request.
///
/// All slice fields (`url`, header names/values) point into the connection's
/// recv buffer. The struct is populated by the parser and consumed by the
/// router and handler layers.
pub const Request = struct {
    /// HTTP method (GET, POST, etc.).
    method: Method,

    /// Raw request URL, e.g. `"/v1/chat/completions?stream=true"`.
    url: []const u8,

    /// HTTP major version number (typically 1).
    version_major: u8,

    /// HTTP minor version number (0 for HTTP/1.0, 1 for HTTP/1.1).
    version_minor: u8,

    /// Parsed headers in the order they appeared in the request.
    headers: []const Header,

    /// Whether the connection should be kept alive after this request.
    /// Derived from the `Connection` header and HTTP version defaults.
    keep_alive: bool,

    /// Value of the `Content-Length` header, if present. `null` means the
    /// header was absent (not the same as zero).
    content_length: ?u64,

    /// `true` when `Transfer-Encoding: chunked` is set.
    chunked: bool,

    /// Look up a header value by name (case-insensitive).
    ///
    /// Returns the value of the first header whose name matches, or `null`
    /// if no such header exists. This is an O(n) scan — fine for the small
    /// number of headers in a typical inference request.
    pub fn header(self: Request, name: []const u8) ?[]const u8 {
        for (self.headers) |h| {
            if (std.ascii.eqlIgnoreCase(h.name, name)) {
                return h.value;
            }
        }
        return null;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Method.fromLlhttp maps standard methods correctly" {
    // Verify every supported code maps to the right variant.
    try std.testing.expectEqual(Method.DELETE, Method.fromLlhttp(0).?);
    try std.testing.expectEqual(Method.GET, Method.fromLlhttp(1).?);
    try std.testing.expectEqual(Method.HEAD, Method.fromLlhttp(2).?);
    try std.testing.expectEqual(Method.POST, Method.fromLlhttp(3).?);
    try std.testing.expectEqual(Method.PUT, Method.fromLlhttp(4).?);
    try std.testing.expectEqual(Method.CONNECT, Method.fromLlhttp(5).?);
    try std.testing.expectEqual(Method.OPTIONS, Method.fromLlhttp(6).?);
    try std.testing.expectEqual(Method.TRACE, Method.fromLlhttp(7).?);
    try std.testing.expectEqual(Method.PATCH, Method.fromLlhttp(28).?);
}

test "Method.fromLlhttp returns null for unknown codes" {
    // Codes outside our supported set (e.g. WebDAV methods) must yield null.
    try std.testing.expect(Method.fromLlhttp(8) == null);
    try std.testing.expect(Method.fromLlhttp(99) == null);
    try std.testing.expect(Method.fromLlhttp(255) == null);
}

test "Request.header finds headers case-insensitively" {
    const headers = [_]Header{
        .{ .name = "Content-Type", .value = "application/json" },
        .{ .name = "Authorization", .value = "Bearer tok_abc" },
        .{ .name = "X-Request-ID", .value = "req-42" },
    };

    const req = Request{
        .method = .POST,
        .url = "/v1/chat/completions",
        .version_major = 1,
        .version_minor = 1,
        .headers = &headers,
        .keep_alive = true,
        .content_length = 128,
        .chunked = false,
    };

    // Exact case.
    try std.testing.expectEqualStrings("application/json", req.header("Content-Type").?);

    // Different case — must still match.
    try std.testing.expectEqualStrings("Bearer tok_abc", req.header("authorization").?);
    try std.testing.expectEqualStrings("req-42", req.header("x-request-id").?);
}

test "Request.header returns null for missing headers" {
    const req = Request{
        .method = .GET,
        .url = "/health",
        .version_major = 1,
        .version_minor = 1,
        .headers = &.{},
        .keep_alive = true,
        .content_length = null,
        .chunked = false,
    };

    try std.testing.expect(req.header("X-Missing") == null);
}
