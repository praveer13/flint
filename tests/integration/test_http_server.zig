//! Integration tests for Flint's HTTP server.
//!
//! These tests exercise the full TCP → HTTP parse → route → response pipeline
//! by starting a real TCP server on an ephemeral port (port 0), connecting as
//! a client, sending raw HTTP requests, and verifying the responses.
//!
//! The tests use `std.testing.io` (`Io.Threaded`) as the I/O backend. This
//! is the same sequential-looking code that runs on `Io.Evented` in production,
//! just backed by a thread pool instead of io_uring.
//!
//! ## Test strategy
//!
//! Each test:
//! 1. Listens on 127.0.0.1:0 (OS picks an ephemeral port)
//! 2. Reads the assigned port from `server.socket.address`
//! 3. Spawns the accept loop as an async fiber via `io.async`
//! 4. Connects a TCP client to localhost:port
//! 5. Sends a raw HTTP request and reads the full response
//! 6. Asserts the response contains expected status codes and body content

const std = @import("std");
const net = std.Io.net;
const Io = std.Io;
const testing = std.testing;

const connection = @import("flint").connection;

/// Starts a server listening on an ephemeral port and returns the server
/// and the port it bound to. The caller must `deinit` the server.
fn startServer(io: Io) !struct { server: net.Server, port: u16 } {
    // Listen on loopback:0 — the OS assigns a free ephemeral port.
    const addr: net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try addr.listen(io, .{ .reuse_address = true });
    const port = server.socket.address.getPort();
    return .{ .server = server, .port = port };
}

/// Accepts a single connection on `server` and handles it. Intended to be
/// spawned as an async fiber so the test can concurrently connect as a client.
fn acceptOne(gpa: std.mem.Allocator, io: Io, server: *net.Server) void {
    const client = server.accept(io) catch return;
    connection.handleConnection(gpa, io, client);
}

/// Connects to localhost:port, sends `request_bytes`, and reads the full
/// response into a dynamically allocated buffer.
fn httpRoundtrip(io: Io, port: u16, request_bytes: []const u8) ![]u8 {
    const addr: net.IpAddress = .{ .ip4 = .loopback(port) };
    const stream = try addr.connect(io, .{ .mode = .stream });
    defer stream.close(io);

    var write_buf: [4096]u8 = undefined;
    var writer = stream.writer(io, &write_buf);
    try writer.interface.writeAll(request_bytes);
    try writer.interface.flush();

    // Signal that we're done sending — this lets the server-side read
    // see EndOfStream after it finishes processing, which is important
    // for non-keep-alive responses and SSE streams that read until EOF.
    try stream.shutdown(io, .send);

    // Read the full response.
    var read_buf: [8192]u8 = undefined;
    var reader = stream.reader(io, &read_buf);
    const resp = try reader.interface.allocRemaining(testing.allocator, .limited(65536));
    return resp;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "GET /health returns 200 OK with JSON status" {
    const io = testing.io;
    const gpa = testing.allocator;

    var result = try startServer(io);
    defer result.server.deinit(io);

    // Spawn accept handler for one connection. We await the future after the
    // roundtrip so the server fiber's resources are freed (no leak).
    var future = io.async(acceptOne, .{ gpa, io, &result.server });

    const resp = try httpRoundtrip(io, result.port,
        "GET /health HTTP/1.1\r\n" ++
        "Host: localhost\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
    );
    defer gpa.free(resp);

    // Wait for the server fiber to finish handling the connection.
    future.await(io);

    // Verify status line.
    try testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);
    // Verify JSON body.
    try testing.expect(std.mem.indexOf(u8, resp, "{\"status\":\"ok\"}") != null);
}

test "GET /v1/models returns 200 with model list JSON" {
    const io = testing.io;
    const gpa = testing.allocator;

    var result = try startServer(io);
    defer result.server.deinit(io);

    var future = io.async(acceptOne, .{ gpa, io, &result.server });

    const resp = try httpRoundtrip(io, result.port,
        "GET /v1/models HTTP/1.1\r\n" ++
        "Host: localhost\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
    );
    defer gpa.free(resp);

    future.await(io);

    try testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);
    try testing.expect(std.mem.indexOf(u8, resp, "\"object\":\"list\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp, "\"id\":\"mock-model\"") != null);
}

test "POST /v1/chat/completions returns SSE stream with mock tokens" {
    const io = testing.io;
    const gpa = testing.allocator;

    var result = try startServer(io);
    defer result.server.deinit(io);

    var future = io.async(acceptOne, .{ gpa, io, &result.server });

    const body = "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    var req_buf: [512]u8 = undefined;
    const request = std.fmt.bufPrint(&req_buf,
        "POST /v1/chat/completions HTTP/1.1\r\n" ++
        "Host: localhost\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{ body.len, body },
    ) catch unreachable;

    const resp = try httpRoundtrip(io, result.port, request);
    defer gpa.free(resp);

    future.await(io);

    // Verify SSE response headers.
    try testing.expect(std.mem.indexOf(u8, resp, "200 OK") != null);
    try testing.expect(std.mem.indexOf(u8, resp, "text/event-stream") != null);

    // Verify SSE events contain mock tokens. The openai handler streams
    // "Hello", "!", " I'm", " Flint", etc.
    try testing.expect(std.mem.indexOf(u8, resp, "data: ") != null);
    try testing.expect(std.mem.indexOf(u8, resp, "\"content\":\"Hello\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp, "\"content\":\" Flint\"") != null);

    // Verify the stream ends with the [DONE] sentinel.
    try testing.expect(std.mem.indexOf(u8, resp, "data: [DONE]") != null);

    // Verify the final chunk carries "finish_reason":"stop".
    try testing.expect(std.mem.indexOf(u8, resp, "\"finish_reason\":\"stop\"") != null);
}

test "GET /unknown returns 404 Not Found" {
    const io = testing.io;
    const gpa = testing.allocator;

    var result = try startServer(io);
    defer result.server.deinit(io);

    var future = io.async(acceptOne, .{ gpa, io, &result.server });

    const resp = try httpRoundtrip(io, result.port,
        "GET /unknown HTTP/1.1\r\n" ++
        "Host: localhost\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
    );
    defer gpa.free(resp);

    future.await(io);

    try testing.expect(std.mem.indexOf(u8, resp, "404") != null);
    try testing.expect(std.mem.indexOf(u8, resp, "Not Found") != null);
}
