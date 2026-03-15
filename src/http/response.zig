//! HTTP response writer and SSE (Server-Sent Events) support.
//!
//! This module provides functions for writing HTTP/1.1 responses and
//! streaming SSE events over a connection's writer. It sits in the
//! protocol layer between the HTTP parser (which reads requests) and
//! the router (which decides what to send back).
//!
//! ## SSE in brief
//!
//! Server-Sent Events is a simple text protocol for server→client
//! streaming over a long-lived HTTP connection. The server sends an
//! initial response with `Content-Type: text/event-stream`, then
//! pushes newline-delimited frames as data becomes available:
//!
//! ```
//! data: {"token": "Hello"}
//!
//! data: {"token": " world"}
//!
//! data: [DONE]
//!
//! ```
//!
//! Each frame is `data: <payload>\n\n` — the double newline signals
//! the end of that event. The client (browser EventSource or curl -N)
//! receives each chunk as it arrives. Flint uses SSE to stream tokens
//! from the LLM to the caller in real time.
//!
//! ## Usage
//!
//! All write functions accept a `*Io.Writer` (the interface pointer
//! obtained from a connection's buffered writer). They flush after
//! every write to ensure bytes reach the client promptly — critical
//! for SSE where latency matters more than throughput.

const std = @import("std");
const Io = std.Io;

/// HTTP response status codes used by Flint.
///
/// Only the codes we actually return are listed here — no need for a
/// full registry. Each variant carries its numeric code as the enum
/// value, making it trivial to format into a status line.
pub const Status = enum(u16) {
    ok = 200,
    bad_request = 400,
    not_found = 404,
    method_not_allowed = 405,
    too_many_requests = 429,
    internal_server_error = 500,

    /// Returns the standard HTTP reason phrase for this status code.
    /// Used when formatting the `HTTP/1.1 <code> <phrase>` status line.
    pub fn phrase(self: Status) []const u8 {
        return switch (self) {
            .ok => "OK",
            .bad_request => "Bad Request",
            .not_found => "Not Found",
            .method_not_allowed => "Method Not Allowed",
            .too_many_requests => "Too Many Requests",
            .internal_server_error => "Internal Server Error",
        };
    }
};

/// Writes a complete HTTP/1.1 response: status line, headers, and body.
///
/// This is the workhorse for non-streaming responses (health checks,
/// error pages, JSON API replies). The connection is kept alive by
/// default — HTTP/1.1 clients assume keep-alive unless told otherwise.
pub fn writeResponse(
    writer: *Io.Writer,
    status: Status,
    content_type: []const u8,
    body_data: []const u8,
) !void {
    // Status line: "HTTP/1.1 200 OK\r\n"
    var status_buf: [64]u8 = undefined;
    const status_line = std.fmt.bufPrint(&status_buf, "HTTP/1.1 {d} {s}\r\n", .{
        @intFromEnum(status),
        status.phrase(),
    }) catch return error.Overflow;
    try writer.writeAll(status_line);

    // Content-Type header
    try writer.writeAll("Content-Type: ");
    try writer.writeAll(content_type);
    try writer.writeAll("\r\n");

    // Content-Length header — lets the client know exactly how many
    // body bytes to expect, enabling connection reuse (keep-alive).
    var len_buf: [64]u8 = undefined;
    const len_str = std.fmt.bufPrint(&len_buf, "Content-Length: {d}\r\n", .{
        body_data.len,
    }) catch return error.Overflow;
    try writer.writeAll(len_str);

    try writer.writeAll("Connection: keep-alive\r\n");

    // Blank line separates headers from body.
    try writer.writeAll("\r\n");

    try writer.writeAll(body_data);
    try writer.flush();
}

/// Writes the SSE response headers and flushes.
///
/// After this call the connection is in "streaming mode" — the caller
/// should follow up with `writeSseEvent` for each token and finish
/// with `writeSseDone`. The headers tell the client:
///
/// - `text/event-stream`: this is an SSE stream, parse frames
/// - `no-cache`: do not buffer in proxies or the browser
/// - `keep-alive`: the connection stays open for streaming
pub fn writeSseHeaders(writer: *Io.Writer) !void {
    try writer.writeAll(
        "HTTP/1.1 200 OK\r\n" ++
            "Content-Type: text/event-stream\r\n" ++
            "Cache-Control: no-cache\r\n" ++
            "Connection: keep-alive\r\n" ++
            "\r\n",
    );
    try writer.flush();
}

/// Writes a single SSE event frame: `data: <payload>\n\n`.
///
/// The caller provides a scratch buffer `buf` for formatting. The
/// payload is typically a JSON object containing a token, e.g.
/// `{"token": "Hello"}`. The double newline at the end is the SSE
/// frame delimiter — the client fires its `onmessage` callback
/// after receiving it.
pub fn writeSseEvent(writer: *Io.Writer, buf: []u8, data: []const u8) !void {
    const frame = std.fmt.bufPrint(buf, "data: {s}\n\n", .{data}) catch
        return error.Overflow;
    try writer.writeAll(frame);
    try writer.flush();
}

/// Writes the SSE termination sentinel: `data: [DONE]\n\n`.
///
/// This is an OpenAI API convention — the `[DONE]` message tells the
/// client that generation is complete and no more tokens will follow.
/// The connection can then be reused for a new request.
pub fn writeSseDone(writer: *Io.Writer) !void {
    try writer.writeAll("data: [DONE]\n\n");
    try writer.flush();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Status enum has correct integer values" {
    const testing = std.testing;

    try testing.expectEqual(@as(u16, 200), @intFromEnum(Status.ok));
    try testing.expectEqual(@as(u16, 400), @intFromEnum(Status.bad_request));
    try testing.expectEqual(@as(u16, 404), @intFromEnum(Status.not_found));
    try testing.expectEqual(@as(u16, 405), @intFromEnum(Status.method_not_allowed));
    try testing.expectEqual(@as(u16, 429), @intFromEnum(Status.too_many_requests));
    try testing.expectEqual(@as(u16, 500), @intFromEnum(Status.internal_server_error));
}

test "Status.phrase returns correct reason strings" {
    const testing = std.testing;

    try testing.expectEqualStrings("OK", Status.ok.phrase());
    try testing.expectEqualStrings("Bad Request", Status.bad_request.phrase());
    try testing.expectEqualStrings("Not Found", Status.not_found.phrase());
    try testing.expectEqualStrings("Method Not Allowed", Status.method_not_allowed.phrase());
    try testing.expectEqualStrings("Too Many Requests", Status.too_many_requests.phrase());
    try testing.expectEqualStrings("Internal Server Error", Status.internal_server_error.phrase());
}
