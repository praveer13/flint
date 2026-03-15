//! HTTP/1.1 request parser — Zig-idiomatic wrapper around llhttp.
//!
//! llhttp is the HTTP parser from Node.js, battle-tested across millions of
//! production deployments. We use it rather than rolling our own because HTTP/1.1
//! parsing has a long tail of edge cases (request smuggling, chunked encoding
//! quirks, header injection) that llhttp has spent years hardening against.
//!
//! This module wraps llhttp's streaming C callback API into Zig-idiomatic types.
//! The key design challenge is the **"dangling slice" problem**: llhttp delivers
//! header data in fragments via callbacks, and we accumulate those fragments into
//! a contiguous `header_data` buffer. If we stored `[]const u8` slices pointing
//! into that buffer as we go, any reallocation during a later append would
//! invalidate earlier slices. Instead, we store `HeaderRange` (start offset +
//! length) during parsing, then resolve ranges to actual slices in
//! `onHeadersComplete` when no more appends will happen.
//!
//! Unlike pike's parser (which supports both request and response modes), this
//! parser is **request-only** — Flint generates responses but never parses them.
//! It also accumulates the full request body into `body_buf` (rather than using
//! a callback) because downstream handlers need the complete JSON body for API
//! parsing.
//!
//! Exported sub-modules:
//! ```
//! const hp = @import("http_parser");
//! const Request = hp.types.Request;
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const types = @import("types.zig");

const c = @cImport({
    @cInclude("llhttp.h");
});

pub const Parser = struct {
    const Self = @This();

    pub const FeedError = error{
        InvalidMethod,
        InvalidUrl,
        InvalidVersion,
        InvalidHeader,
        InvalidContentLength,
        InvalidChunkSize,
        InvalidStatus,
        InvalidEofState,
        InvalidTransferEncoding,
        CallbackError,
        ParseError,
    };

    // --- Parsed results ---

    /// Populated when `on_headers_complete` fires. Contains method, URL,
    /// headers, and connection metadata. Valid until `reset()` or `deinit()`.
    request: ?types.Request,

    /// True after `on_message_complete` fires (full message including body).
    message_complete: bool,

    /// True after `on_headers_complete` fires (headers fully parsed).
    headers_complete: bool,

    // --- Internal state ---

    /// The underlying llhttp C parser instance.
    c_parser: c.llhttp_t,

    /// llhttp callback settings. Must outlive the parser (llhttp stores a
    /// pointer to this). We keep it as a field so the lifetime is automatic.
    c_settings: c.llhttp_settings_t,

    /// Allocator used for all dynamic buffers (url_buf, header_data, etc.).
    allocator: Allocator,

    /// Accumulates URL fragments from potentially multiple `on_url` callbacks.
    /// llhttp may deliver the URL in pieces if the input buffer is split across
    /// multiple `feed()` calls.
    url_buf: std.ArrayListUnmanaged(u8),

    /// Accumulates header field name fragments from `on_header_field` callbacks.
    field_buf: std.ArrayListUnmanaged(u8),

    /// Accumulates header value fragments from `on_header_value` callbacks.
    value_buf: std.ArrayListUnmanaged(u8),

    /// Contiguous storage for all header name/value bytes. Individual headers
    /// are identified by `HeaderRange` offsets into this buffer. We append here
    /// in `on_header_field_complete` and `on_header_value_complete`, then
    /// resolve ranges to slices in `on_headers_complete`.
    header_data: std.ArrayListUnmanaged(u8),

    /// Offset/length pairs into `header_data` for each header's name and value.
    /// We store ranges (not slices) to avoid dangling pointers when
    /// `header_data` reallocates. Resolved to `[]const types.Header` in
    /// `on_headers_complete`.
    header_ranges: std.ArrayListUnmanaged(HeaderRange),

    /// Accumulates request body bytes from `on_body` callbacks. Unlike pike
    /// (which forwards body chunks via a callback), we buffer the entire body
    /// because Flint's API handlers need the complete JSON payload for parsing.
    body_buf: std.ArrayListUnmanaged(u8),

    /// The resolved headers slice, allocated in `onHeadersComplete`. Owned by
    /// the parser and freed in `deinit` / `reset`.
    resolved_headers: ?[]types.Header,

    /// The resolved URL string, copied from `url_buf` in `onUrlComplete`.
    /// Owned by the parser and freed in `deinit` / `reset`.
    resolved_url: ?[]u8,

    const HeaderRange = struct {
        name_start: usize,
        name_len: usize,
        value_start: usize,
        value_len: usize,
    };

    /// Initialize a new request parser.
    ///
    /// The parser is ready to receive data via `feed()`. The caller owns the
    /// returned value and must call `deinit()` when done.
    ///
    /// Note: we do NOT set `c_parser.data` here because `init` returns by value
    /// — the struct address changes when it lands at the caller's site. The
    /// context pointer is set in `feed()` where `self` has its final address.
    pub fn init(allocator: Allocator) Parser {
        var self: Parser = .{
            .request = null,
            .message_complete = false,
            .headers_complete = false,
            .c_parser = undefined,
            .c_settings = undefined,
            .allocator = allocator,
            .url_buf = .empty,
            .field_buf = .empty,
            .value_buf = .empty,
            .header_data = .empty,
            .header_ranges = .empty,
            .body_buf = .empty,
            .resolved_headers = null,
            .resolved_url = null,
        };

        // Initialize llhttp settings with our callback functions.
        c.llhttp_settings_init(&self.c_settings);
        self.c_settings.on_url = onUrl;
        self.c_settings.on_url_complete = onUrlComplete;
        self.c_settings.on_header_field = onHeaderField;
        self.c_settings.on_header_field_complete = onHeaderFieldComplete;
        self.c_settings.on_header_value = onHeaderValue;
        self.c_settings.on_header_value_complete = onHeaderValueComplete;
        self.c_settings.on_headers_complete = onHeadersComplete;
        self.c_settings.on_body = onBody;
        self.c_settings.on_message_complete = onMessageComplete;

        // Always request mode — Flint only parses incoming requests.
        c.llhttp_init(&self.c_parser, c.HTTP_REQUEST, &self.c_settings);

        return self;
    }

    /// Release all resources owned by the parser.
    pub fn deinit(self: *Self) void {
        self.url_buf.deinit(self.allocator);
        self.field_buf.deinit(self.allocator);
        self.value_buf.deinit(self.allocator);
        self.header_data.deinit(self.allocator);
        self.header_ranges.deinit(self.allocator);
        self.body_buf.deinit(self.allocator);
        if (self.resolved_headers) |hdrs| self.allocator.free(hdrs);
        if (self.resolved_url) |url| self.allocator.free(url);
    }

    /// Feed a chunk of HTTP data to the parser. Returns the number of bytes
    /// consumed, which may be less than `data.len` if the parser encounters
    /// the end of a message (e.g., pipelined requests).
    ///
    /// On success the parser's `request`, `headers_complete`, and
    /// `message_complete` fields reflect the parse state. Call `feed()` again
    /// with remaining data to continue parsing (e.g., for pipelined messages).
    ///
    /// **Why we reset `c_parser.data` and `c_parser.settings` on every call:**
    /// `init()` returns the Parser by value, so the struct moves to wherever
    /// the caller stores it. The `data` and `settings` pointers that were set
    /// inside `init()` point to the *old* (now-invalid) address. We patch them
    /// here where `self` has its stable, final address.
    pub fn feed(self: *Self, data: []const u8) FeedError!usize {
        self.c_parser.data = @ptrCast(self);
        self.c_parser.settings = @ptrCast(&self.c_settings);

        // llhttp_execute returns llhttp_errno_t, which Zig's @cImport maps to
        // c_uint. We compare directly against the HPE_* constants.
        const rc: c_uint = c.llhttp_execute(
            &self.c_parser,
            @ptrCast(data.ptr),
            data.len,
        );

        if (rc != c.HPE_OK) {
            // HPE_PAUSED_UPGRADE is not an error — it means the parser saw an
            // Upgrade header and paused after the headers. The caller can
            // resume if they want.
            if (rc == c.HPE_PAUSED_UPGRADE) {
                c.llhttp_resume_after_upgrade(&self.c_parser);
                const err_pos = c.llhttp_get_error_pos(&self.c_parser);
                const consumed = @intFromPtr(err_pos) - @intFromPtr(data.ptr);
                return consumed;
            }
            return mapError(rc);
        }

        return data.len;
    }

    /// Return the accumulated request body. This is the complete body received
    /// so far — for a fully parsed message (`message_complete == true`), this
    /// is the entire request body.
    ///
    /// The returned slice is valid until `reset()` or `deinit()` is called.
    pub fn body(self: *Self) []const u8 {
        return self.body_buf.items;
    }

    /// Reset the parser for keep-alive reuse. Clears all parsed state but
    /// preserves allocated buffer capacity to avoid re-allocation on the next
    /// message.
    pub fn reset(self: *Self) void {
        // Free resolved data from the previous message.
        if (self.resolved_headers) |hdrs| {
            self.allocator.free(hdrs);
            self.resolved_headers = null;
        }
        if (self.resolved_url) |url| {
            self.allocator.free(url);
            self.resolved_url = null;
        }

        // Clear accumulation buffers but keep capacity.
        self.url_buf.clearRetainingCapacity();
        self.field_buf.clearRetainingCapacity();
        self.value_buf.clearRetainingCapacity();
        self.header_data.clearRetainingCapacity();
        self.header_ranges.clearRetainingCapacity();
        self.body_buf.clearRetainingCapacity();

        // Clear result fields.
        self.request = null;
        self.headers_complete = false;
        self.message_complete = false;

        // Reset llhttp state. This preserves type, settings, and lenient flags.
        c.llhttp_reset(&self.c_parser);
    }

    // =========================================================================
    // C callbacks
    // =========================================================================
    //
    // llhttp is a C library that communicates parse events through function
    // pointers (the "settings" callbacks). Each callback receives a pointer
    // to the C `llhttp_t` struct. We recover our owning `Parser` instance
    // via the `data` field we set in `feed()`.
    //
    // All callbacks must use `callconv(.c)` because they're called from C
    // code. They return 0 on success or -1 on error (which causes
    // llhttp_execute to return HPE_CB_*).

    /// Recover the owning `Parser` from a C parser pointer.
    fn getSelf(p: ?*c.llhttp_t) *Self {
        const ptr = p.?;
        return @ptrCast(@alignCast(ptr.data));
    }

    /// Accumulate URL fragment. llhttp may deliver the URL across multiple
    /// callbacks if the input is split.
    fn onUrl(p: ?*c.llhttp_t, at: [*c]const u8, len: usize) callconv(.c) c_int {
        const self = getSelf(p);
        self.url_buf.appendSlice(self.allocator, at[0..len]) catch return -1;
        return 0;
    }

    /// URL fully received — copy the accumulated bytes into `resolved_url` so
    /// the `url_buf` can be reused.
    fn onUrlComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        const url = self.allocator.dupe(u8, self.url_buf.items) catch return -1;
        if (self.resolved_url) |old| self.allocator.free(old);
        self.resolved_url = url;
        self.url_buf.clearRetainingCapacity();
        return 0;
    }

    /// Accumulate header field name fragment.
    fn onHeaderField(p: ?*c.llhttp_t, at: [*c]const u8, len: usize) callconv(.c) c_int {
        const self = getSelf(p);
        self.field_buf.appendSlice(self.allocator, at[0..len]) catch return -1;
        return 0;
    }

    /// Header field name fully received — append the name bytes into
    /// `header_data` and record the start position. The `HeaderRange` is
    /// partially filled here; the value half is filled in
    /// `onHeaderValueComplete`.
    fn onHeaderFieldComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        const start = self.header_data.items.len;
        self.header_data.appendSlice(self.allocator, self.field_buf.items) catch return -1;
        self.header_ranges.append(self.allocator, .{
            .name_start = start,
            .name_len = self.field_buf.items.len,
            .value_start = 0,
            .value_len = 0,
        }) catch return -1;
        self.field_buf.clearRetainingCapacity();
        return 0;
    }

    /// Accumulate header value fragment.
    fn onHeaderValue(p: ?*c.llhttp_t, at: [*c]const u8, len: usize) callconv(.c) c_int {
        const self = getSelf(p);
        self.value_buf.appendSlice(self.allocator, at[0..len]) catch return -1;
        return 0;
    }

    /// Header value fully received — append value bytes into `header_data` and
    /// complete the current `HeaderRange` with value position.
    fn onHeaderValueComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        const start = self.header_data.items.len;
        self.header_data.appendSlice(self.allocator, self.value_buf.items) catch return -1;

        // The last range was partially filled by onHeaderFieldComplete.
        const range = &self.header_ranges.items[self.header_ranges.items.len - 1];
        range.value_start = start;
        range.value_len = self.value_buf.items.len;

        self.value_buf.clearRetainingCapacity();
        return 0;
    }

    /// All headers received — resolve `HeaderRange` offsets into actual slices
    /// and assemble the `Request` struct.
    ///
    /// At this point `header_data` is frozen (no more appends), so slices into
    /// it are safe — the dangling-slice problem is avoided.
    fn onHeadersComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        const parser = &p.?.*;

        // Resolve header ranges to slices.
        const headers = self.allocator.alloc(types.Header, self.header_ranges.items.len) catch return -1;
        for (self.header_ranges.items, 0..) |range, i| {
            headers[i] = .{
                .name = self.header_data.items[range.name_start..][0..range.name_len],
                .value = self.header_data.items[range.value_start..][0..range.value_len],
            };
        }

        // Free any previously resolved headers (shouldn't happen in normal
        // flow, but be safe).
        if (self.resolved_headers) |old| self.allocator.free(old);
        self.resolved_headers = headers;

        const keep_alive = c.llhttp_should_keep_alive(&self.c_parser) != 0;
        const flags = parser.flags;
        const chunked = (flags & c.F_CHUNKED) != 0;
        const has_content_length = (flags & c.F_CONTENT_LENGTH) != 0;
        const content_length: ?u64 = if (has_content_length) parser.content_length else null;

        const method = types.Method.fromLlhttp(parser.method) orelse return -1;
        self.request = .{
            .method = method,
            .url = self.resolved_url orelse "",
            .version_major = parser.http_major,
            .version_minor = parser.http_minor,
            .headers = headers,
            .keep_alive = keep_alive,
            .content_length = content_length,
            .chunked = chunked,
        };

        self.headers_complete = true;
        return 0;
    }

    /// Body data received — accumulate into `body_buf`. Unlike pike (which
    /// forwards body chunks via a callback), we buffer the full body because
    /// Flint needs the complete JSON payload for API request parsing.
    fn onBody(p: ?*c.llhttp_t, at: [*c]const u8, len: usize) callconv(.c) c_int {
        const self = getSelf(p);
        self.body_buf.appendSlice(self.allocator, at[0..len]) catch return -1;
        return 0;
    }

    /// Full message received (headers + body).
    fn onMessageComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        self.message_complete = true;
        return 0;
    }

    // =========================================================================
    // Error mapping
    // =========================================================================

    /// Map an llhttp error code to a Zig error. We map specific error codes to
    /// specific errors so callers can handle them precisely; everything else
    /// becomes `ParseError`.
    fn mapError(err: c_uint) FeedError {
        return switch (err) {
            c.HPE_INVALID_METHOD => error.InvalidMethod,
            c.HPE_INVALID_URL => error.InvalidUrl,
            c.HPE_INVALID_VERSION, c.HPE_INVALID_CONSTANT => error.InvalidVersion,
            c.HPE_INVALID_HEADER_TOKEN => error.InvalidHeader,
            c.HPE_INVALID_CONTENT_LENGTH, c.HPE_UNEXPECTED_CONTENT_LENGTH => error.InvalidContentLength,
            c.HPE_INVALID_CHUNK_SIZE => error.InvalidChunkSize,
            c.HPE_INVALID_STATUS => error.InvalidStatus,
            c.HPE_INVALID_EOF_STATE => error.InvalidEofState,
            c.HPE_INVALID_TRANSFER_ENCODING => error.InvalidTransferEncoding,
            c.HPE_CB_MESSAGE_BEGIN,
            c.HPE_CB_HEADERS_COMPLETE,
            c.HPE_CB_MESSAGE_COMPLETE,
            c.HPE_CB_URL_COMPLETE,
            c.HPE_CB_HEADER_FIELD_COMPLETE,
            c.HPE_CB_HEADER_VALUE_COMPLETE,
            => error.CallbackError,
            else => error.ParseError,
        };
    }
};

// =============================================================================
// Tests
// =============================================================================

test {
    // Pull in types.zig tests so they're discovered via the http_parser module.
    _ = @import("types.zig");
}

test "parse complete GET request — verify method, url, message_complete" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    const input = "GET /v1/models HTTP/1.1\r\nHost: localhost:8080\r\nAccept: application/json\r\n\r\n";
    _ = try parser.feed(input);

    try std.testing.expect(parser.headers_complete);
    try std.testing.expect(parser.message_complete);

    const req = parser.request.?;
    try std.testing.expectEqual(types.Method.GET, req.method);
    try std.testing.expectEqualStrings("/v1/models", req.url);
    try std.testing.expectEqual(@as(u8, 1), req.version_major);
    try std.testing.expectEqual(@as(u8, 1), req.version_minor);
    try std.testing.expectEqual(@as(usize, 2), req.headers.len);
    try std.testing.expectEqualStrings("Host", req.headers[0].name);
    try std.testing.expectEqualStrings("localhost:8080", req.headers[0].value);
    try std.testing.expect(req.keep_alive);
}

test "parse POST with JSON body — verify url, body(), content_length, message_complete" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    const json_body = "{\"model\":\"llama-3\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    const input = "POST /v1/chat/completions HTTP/1.1\r\n" ++
        "Host: localhost:8080\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: " ++ std.fmt.comptimePrint("{d}", .{json_body.len}) ++ "\r\n" ++
        "\r\n" ++
        json_body;

    _ = try parser.feed(input);

    try std.testing.expect(parser.message_complete);

    const req = parser.request.?;
    try std.testing.expectEqual(types.Method.POST, req.method);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.url);
    try std.testing.expectEqual(@as(?u64, json_body.len), req.content_length);
    try std.testing.expect(!req.chunked);
    try std.testing.expectEqualStrings(json_body, parser.body());
}

test "parse request split across multiple feeds — verify URL reassembled correctly" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    // Split the request in the middle of the URL.
    _ = try parser.feed("GET /v1/chat/com");
    try std.testing.expect(parser.request == null);
    try std.testing.expect(!parser.headers_complete);

    _ = try parser.feed("pletions HTTP/1.1\r\nHost: localhost\r\n\r\n");

    try std.testing.expect(parser.message_complete);
    const req = parser.request.?;
    try std.testing.expectEqualStrings("/v1/chat/completions", req.url);
    try std.testing.expectEqual(types.Method.GET, req.method);
}

test "invalid method returns error" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    const result = parser.feed("FOOBAR / HTTP/1.1\r\n\r\n");
    try std.testing.expectError(error.InvalidMethod, result);
}

test "reset and reuse for keep-alive — parse two requests on same parser" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    // First request: GET
    _ = try parser.feed("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    try std.testing.expectEqualStrings("/health", parser.request.?.url);
    try std.testing.expectEqual(types.Method.GET, parser.request.?.method);
    try std.testing.expect(parser.message_complete);
    try std.testing.expect(parser.request.?.keep_alive);

    // Reset for keep-alive reuse.
    parser.reset();
    try std.testing.expect(parser.request == null);
    try std.testing.expect(!parser.message_complete);
    try std.testing.expect(!parser.headers_complete);
    try std.testing.expectEqual(@as(usize, 0), parser.body().len);

    // Second request: POST with body
    const body2 = "{\"prompt\":\"hello\"}";
    const req2 = "POST /v1/completions HTTP/1.1\r\n" ++
        "Host: localhost\r\n" ++
        "Content-Length: " ++ std.fmt.comptimePrint("{d}", .{body2.len}) ++ "\r\n" ++
        "\r\n" ++
        body2;
    _ = try parser.feed(req2);

    try std.testing.expect(parser.message_complete);
    try std.testing.expectEqualStrings("/v1/completions", parser.request.?.url);
    try std.testing.expectEqual(types.Method.POST, parser.request.?.method);
    try std.testing.expectEqualStrings(body2, parser.body());
}

test "HTTP/1.0 without Connection header implies no keep-alive" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    _ = try parser.feed("GET / HTTP/1.0\r\nHost: example.com\r\n\r\n");
    const req = parser.request.?;
    try std.testing.expect(!req.keep_alive);
}

test "Connection: close disables keep-alive on HTTP/1.1" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    _ = try parser.feed("GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    try std.testing.expect(!parser.request.?.keep_alive);
}

test "parse request with many headers" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    const input =
        "GET / HTTP/1.1\r\n" ++
        "Host: example.com\r\n" ++
        "Accept: text/html\r\n" ++
        "Accept-Language: en-US\r\n" ++
        "Accept-Encoding: gzip, deflate\r\n" ++
        "Connection: keep-alive\r\n" ++
        "\r\n";
    _ = try parser.feed(input);

    const req = parser.request.?;
    try std.testing.expectEqual(@as(usize, 5), req.headers.len);
    try std.testing.expectEqualStrings("Host", req.headers[0].name);
    try std.testing.expectEqualStrings("Connection", req.headers[4].name);
    try std.testing.expectEqualStrings("keep-alive", req.headers[4].value);
}

test "body() returns empty slice for bodiless GET" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    _ = try parser.feed("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
    try std.testing.expect(parser.message_complete);
    try std.testing.expectEqual(@as(usize, 0), parser.body().len);
}
