# Phase 1: TCP Server + HTTP Parser + SSE Echo — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working HTTP server using `std.Io` + llhttp that serves health checks and mock SSE streams from an OpenAI-compatible API.

**Architecture:** TCP accept loop spawns fibers per connection (pike pattern). llhttp parses requests. Router dispatches to handlers. SSE writer streams mock tokens. No GPU, no Python, no shared memory.

**Tech Stack:** Zig 0.16, llhttp v9.3.1 (C FFI), std.Io (Evented/Threaded)

**Reference:** Pike project at `~/workplace/pike` — proven working patterns for all std.Io and llhttp usage.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `build.zig.zon`
- Create: `build.zig`
- Create: `src/main.zig` (minimal stub)

**Step 1: Create `build.zig.zon`**

Use pike's llhttp dependency. No TLS needed for Phase 1.

```zig
.{
    .name = .flint,
    .version = "0.0.0",
    .dependencies = .{
        .llhttp = .{
            .url = "https://github.com/nodejs/llhttp/archive/refs/tags/release/v9.3.1.tar.gz",
            .hash = "N-V-__8AABH_BQCTlwSpr4H19z_8_fTfeWLsbGmPL_fNtshb",
        },
    },
    .fingerprint = 0xa1b2c3d4e5f60718,
    .minimum_zig_version = "0.16.0-dev.2694+74f361a5c",
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
    },
}
```

Note: `fingerprint` must be a unique u64. Generate one or let `zig build` tell you.

**Step 2: Create `build.zig`**

Follow pike's module pattern. Create:
- `http_parser` named module (llhttp C + Zig wrapper)
- Main executable importing `http_parser`
- Test step

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const llhttp_dep = b.dependency("llhttp", .{});

    // HTTP parser module — wraps llhttp with Zig API
    const http_parser_module = b.createModule(.{
        .root_source_file = b.path("src/http/parser.zig"),
        .target = target,
        .optimize = optimize,
    });
    http_parser_module.addCSourceFiles(.{
        .root = llhttp_dep.path("src"),
        .files = &.{ "llhttp.c", "api.c", "http.c" },
    });
    http_parser_module.addIncludePath(llhttp_dep.path("include"));
    http_parser_module.link_libc = true;

    // Main executable
    const exe = b.addExecutable(.{
        .name = "flint",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "http_parser", .module = http_parser_module },
            },
        }),
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run flint server");
    run_step.dependOn(&run_cmd.step);

    // Tests — from main module (discovers all referenced test blocks)
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_exe_tests.step);
}
```

**Step 3: Create minimal `src/main.zig` stub**

```zig
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    _ = io;
    std.log.info("flint starting", .{});
}
```

**Step 4: Create directory structure**

```bash
mkdir -p src/http src/net src/api
```

**Step 5: Verify it builds**

```bash
zig build
```

Expected: Builds successfully (may need to fetch llhttp dependency).

**Step 6: Commit**

```bash
git init
git add build.zig build.zig.zon src/main.zig CLAUDE.md docs/
git commit -m "feat: project scaffolding with llhttp dependency"
```

---

### Task 2: HTTP Types

**Files:**
- Create: `src/http/types.zig`

**Step 1: Write types with tests**

Adapt from pike's `src/http/types.zig`. We only need request types (no response parsing — we generate responses, not parse them).

```zig
const std = @import("std");

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

pub const Header = struct {
    name: []const u8,
    value: []const u8,
};

pub const Request = struct {
    method: Method,
    url: []const u8,
    version_major: u8,
    version_minor: u8,
    headers: []const Header,
    keep_alive: bool,
    content_length: ?u64,
    chunked: bool,

    /// Find a header by name (case-insensitive).
    pub fn header(self: Request, name: []const u8) ?[]const u8 {
        for (self.headers) |h| {
            if (std.ascii.eqlIgnoreCase(h.name, name)) return h.value;
        }
        return null;
    }
};

test "Method.fromLlhttp maps standard methods" {
    try std.testing.expectEqual(Method.GET, Method.fromLlhttp(1).?);
    try std.testing.expectEqual(Method.POST, Method.fromLlhttp(3).?);
    try std.testing.expectEqual(Method.PATCH, Method.fromLlhttp(28).?);
    try std.testing.expect(Method.fromLlhttp(99) == null);
}

test "Request.header finds header case-insensitively" {
    const headers = [_]Header{
        .{ .name = "Content-Type", .value = "application/json" },
        .{ .name = "Host", .value = "localhost" },
    };
    const req = Request{
        .method = .POST,
        .url = "/test",
        .version_major = 1,
        .version_minor = 1,
        .headers = &headers,
        .keep_alive = true,
        .content_length = null,
        .chunked = false,
    };
    try std.testing.expectEqualStrings("application/json", req.header("content-type").?);
    try std.testing.expect(req.header("X-Missing") == null);
}
```

**Step 2: Run tests**

```bash
zig build test
```

Expected: PASS. (This requires parser.zig to exist — create an empty placeholder first if needed, or wire types.zig tests separately.)

Note: Since `build.zig` roots tests at `src/main.zig`, we need `main.zig` to reference `http_parser` which references `types.zig`. If the parser isn't ready yet, create `src/http/parser.zig` as a minimal placeholder that imports types:

```zig
const types = @import("types.zig");
test { _ = types; }
```

**Step 3: Commit**

```bash
git add src/http/types.zig src/http/parser.zig
git commit -m "feat: HTTP types (Method, Header, Request)"
```

---

### Task 3: HTTP Parser (llhttp wrapper)

**Files:**
- Modify: `src/http/parser.zig`

**Step 1: Write the parser**

Copy pike's `src/http/parser.zig` and adapt:
- Keep request parsing (remove response parsing — we don't parse responses)
- Add body accumulation (pike uses a callback; we need the full body for JSON parsing)
- Keep all the same llhttp callback patterns

Key differences from pike:
- We accumulate the body into a buffer (need it for JSON parsing in the API layer)
- We only parse requests, never responses
- We add a `body()` method that returns accumulated body bytes

```zig
// src/http/parser.zig
//
// HTTP/1.1 request parser — thin Zig wrapper over llhttp.
// Adapted from pike's parser. See pike/src/http/parser.zig for the original.
//
// Key difference from pike: we accumulate the request body (for JSON API parsing)
// rather than forwarding it via callback. This is fine for an API server where
// request bodies are small JSON payloads (typically <64KB).

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
    request: ?types.Request,
    message_complete: bool,
    headers_complete: bool,

    // --- Internal state ---
    c_parser: c.llhttp_t,
    c_settings: c.llhttp_settings_t,
    allocator: Allocator,

    url_buf: std.ArrayListUnmanaged(u8),
    field_buf: std.ArrayListUnmanaged(u8),
    value_buf: std.ArrayListUnmanaged(u8),
    header_data: std.ArrayListUnmanaged(u8),
    header_ranges: std.ArrayListUnmanaged(HeaderRange),
    body_buf: std.ArrayListUnmanaged(u8),

    resolved_headers: ?[]types.Header,
    resolved_url: ?[]u8,

    const HeaderRange = struct {
        name_start: usize,
        name_len: usize,
        value_start: usize,
        value_len: usize,
    };

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

        c.llhttp_init(&self.c_parser, c.HTTP_REQUEST, &self.c_settings);

        return self;
    }

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

    pub fn feed(self: *Self, data: []const u8) FeedError!usize {
        self.c_parser.data = @ptrCast(self);
        self.c_parser.settings = @ptrCast(&self.c_settings);

        const rc: c_uint = c.llhttp_execute(
            &self.c_parser,
            @ptrCast(data.ptr),
            data.len,
        );

        if (rc != c.HPE_OK) {
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

    /// Returns the accumulated request body, or empty slice if no body.
    pub fn body(self: *Self) []const u8 {
        return self.body_buf.items;
    }

    pub fn reset(self: *Self) void {
        if (self.resolved_headers) |hdrs| {
            self.allocator.free(hdrs);
            self.resolved_headers = null;
        }
        if (self.resolved_url) |url| {
            self.allocator.free(url);
            self.resolved_url = null;
        }

        self.url_buf.clearRetainingCapacity();
        self.field_buf.clearRetainingCapacity();
        self.value_buf.clearRetainingCapacity();
        self.header_data.clearRetainingCapacity();
        self.header_ranges.clearRetainingCapacity();
        self.body_buf.clearRetainingCapacity();

        self.request = null;
        self.headers_complete = false;
        self.message_complete = false;

        c.llhttp_reset(&self.c_parser);
    }

    // --- C callbacks ---

    fn getSelf(p: ?*c.llhttp_t) *Self {
        const ptr = p.?;
        return @ptrCast(@alignCast(ptr.data));
    }

    fn onUrl(p: ?*c.llhttp_t, at: [*c]const u8, len: usize) callconv(.c) c_int {
        const self = getSelf(p);
        self.url_buf.appendSlice(self.allocator, at[0..len]) catch return -1;
        return 0;
    }

    fn onUrlComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        const url = self.allocator.dupe(u8, self.url_buf.items) catch return -1;
        if (self.resolved_url) |old| self.allocator.free(old);
        self.resolved_url = url;
        self.url_buf.clearRetainingCapacity();
        return 0;
    }

    fn onHeaderField(p: ?*c.llhttp_t, at: [*c]const u8, len: usize) callconv(.c) c_int {
        const self = getSelf(p);
        self.field_buf.appendSlice(self.allocator, at[0..len]) catch return -1;
        return 0;
    }

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

    fn onHeaderValue(p: ?*c.llhttp_t, at: [*c]const u8, len: usize) callconv(.c) c_int {
        const self = getSelf(p);
        self.value_buf.appendSlice(self.allocator, at[0..len]) catch return -1;
        return 0;
    }

    fn onHeaderValueComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        const start = self.header_data.items.len;
        self.header_data.appendSlice(self.allocator, self.value_buf.items) catch return -1;

        const range = &self.header_ranges.items[self.header_ranges.items.len - 1];
        range.value_start = start;
        range.value_len = self.value_buf.items.len;

        self.value_buf.clearRetainingCapacity();
        return 0;
    }

    fn onHeadersComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        const parser = &p.?.*;

        const headers = self.allocator.alloc(types.Header, self.header_ranges.items.len) catch return -1;
        for (self.header_ranges.items, 0..) |range, i| {
            headers[i] = .{
                .name = self.header_data.items[range.name_start..][0..range.name_len],
                .value = self.header_data.items[range.value_start..][0..range.value_len],
            };
        }

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

    fn onBody(p: ?*c.llhttp_t, at: [*c]const u8, len: usize) callconv(.c) c_int {
        const self = getSelf(p);
        self.body_buf.appendSlice(self.allocator, at[0..len]) catch return -1;
        return 0;
    }

    fn onMessageComplete(p: ?*c.llhttp_t) callconv(.c) c_int {
        const self = getSelf(p);
        self.message_complete = true;
        return 0;
    }

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

// Pull in types.zig tests
test {
    _ = @import("types.zig");
}

test "parse complete GET request" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    _ = try parser.feed("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");

    const req = parser.request.?;
    try std.testing.expectEqual(types.Method.GET, req.method);
    try std.testing.expectEqualStrings("/health", req.url);
    try std.testing.expect(parser.message_complete);
}

test "parse POST with JSON body" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    const body_str = "{\"model\":\"test\",\"messages\":[]}";
    const input = std.fmt.comptimePrint(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}",
        .{ body_str.len, body_str },
    );
    _ = try parser.feed(input);

    const req = parser.request.?;
    try std.testing.expectEqual(types.Method.POST, req.method);
    try std.testing.expectEqualStrings("/v1/chat/completions", req.url);
    try std.testing.expectEqualStrings(body_str, parser.body());
    try std.testing.expect(parser.message_complete);
}

test "parse split across feeds" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    _ = try parser.feed("GET /api/mod");
    try std.testing.expect(parser.request == null);

    _ = try parser.feed("els HTTP/1.1\r\nHost: localhost\r\n\r\n");
    try std.testing.expectEqualStrings("/api/models", parser.request.?.url);
}

test "invalid method returns error" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    try std.testing.expectError(error.InvalidMethod, parser.feed("FOOBAR / HTTP/1.1\r\n\r\n"));
}

test "reset and reuse for keep-alive" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);
    defer parser.deinit();

    _ = try parser.feed("GET /first HTTP/1.1\r\nHost: localhost\r\n\r\n");
    try std.testing.expectEqualStrings("/first", parser.request.?.url);

    parser.reset();

    _ = try parser.feed("GET /second HTTP/1.1\r\nHost: localhost\r\n\r\n");
    try std.testing.expectEqualStrings("/second", parser.request.?.url);
}
```

**Step 2: Update `src/main.zig` to reference http_parser for test discovery**

```zig
const std = @import("std");

test {
    _ = @import("http_parser");
}

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.log.info("flint starting", .{});
}
```

**Step 3: Run tests**

```bash
zig build test
```

Expected: All tests pass.

**Step 4: Commit**

```bash
git add src/http/parser.zig src/http/types.zig src/main.zig
git commit -m "feat: HTTP request parser wrapping llhttp"
```

---

### Task 4: HTTP Response Writer Utilities

**Files:**
- Create: `src/http/response.zig`

**Step 1: Write response helpers with tests**

We need functions to write HTTP responses. Unlike pike (which forwards raw bytes), we generate our own responses. These work with `Io.Writer` interface pointers.

```zig
const std = @import("std");
const Io = std.Io;

pub const Status = enum(u16) {
    ok = 200,
    bad_request = 400,
    not_found = 404,
    method_not_allowed = 405,
    too_many_requests = 429,
    internal_server_error = 500,

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

/// Write a complete HTTP response (headers + body) and flush.
pub fn writeResponse(
    writer: *Io.Writer,
    status: Status,
    content_type: []const u8,
    body_data: []const u8,
) !void {
    var hdr_buf: [512]u8 = undefined;
    const header = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{
        @intFromEnum(status),
        status.phrase(),
        content_type,
        body_data.len,
    }) catch return error.Overflow;
    try writer.writeAll(header);
    if (body_data.len > 0) {
        try writer.writeAll(body_data);
    }
    try writer.flush();
}

/// Write SSE response headers and flush (start of streaming).
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

/// Write a single SSE data event and flush.
pub fn writeSseEvent(writer: *Io.Writer, buf: []u8, data: []const u8) !void {
    const frame = std.fmt.bufPrint(buf, "data: {s}\n\n", .{data}) catch return error.Overflow;
    try writer.writeAll(frame);
    try writer.flush();
}

/// Write the SSE [DONE] terminator and flush.
pub fn writeSseDone(writer: *Io.Writer) !void {
    try writer.writeAll("data: [DONE]\n\n");
    try writer.flush();
}
```

Note: We can't easily unit-test `Io.Writer` without a real Io backend. These will be tested via integration tests in Task 7.

**Step 2: Commit**

```bash
git add src/http/response.zig
git commit -m "feat: HTTP response writer with SSE support"
```

---

### Task 5: OpenAI API Router

**Files:**
- Create: `src/api/router.zig`
- Create: `src/api/openai.zig`

**Step 1: Write the router**

The router matches parsed requests to handler functions.

```zig
// src/api/router.zig
//
// Routes parsed HTTP requests to handler functions.
// Each handler receives the parsed request, body bytes, and an Io.Writer
// to write the response.

const std = @import("std");
const Io = std.Io;
const http_types = @import("http_parser").types;
const response = @import("../http/response.zig");
const openai = @import("openai.zig");

pub fn route(
    req: http_types.Request,
    body_data: []const u8,
    writer: *Io.Writer,
) !void {
    if (std.mem.eql(u8, req.url, "/health")) {
        return handleHealth(req, writer);
    } else if (std.mem.eql(u8, req.url, "/v1/models")) {
        return handleModels(req, writer);
    } else if (std.mem.eql(u8, req.url, "/v1/chat/completions")) {
        return handleChatCompletions(req, body_data, writer);
    } else {
        return response.writeResponse(writer, .not_found, "text/plain", "Not Found");
    }
}

fn handleHealth(req: http_types.Request, writer: *Io.Writer) !void {
    if (req.method != .GET) {
        return response.writeResponse(writer, .method_not_allowed, "text/plain", "Method Not Allowed");
    }
    return response.writeResponse(writer, .ok, "application/json", "{\"status\":\"ok\"}");
}

fn handleModels(req: http_types.Request, writer: *Io.Writer) !void {
    if (req.method != .GET) {
        return response.writeResponse(writer, .method_not_allowed, "text/plain", "Method Not Allowed");
    }
    const body =
        \\{"object":"list","data":[{"id":"mock-model","object":"model","owned_by":"flint"}]}
    ;
    return response.writeResponse(writer, .ok, "application/json", body);
}

fn handleChatCompletions(
    req: http_types.Request,
    body_data: []const u8,
    writer: *Io.Writer,
) !void {
    if (req.method != .POST) {
        return response.writeResponse(writer, .method_not_allowed, "text/plain", "Method Not Allowed");
    }
    return openai.handleChatCompletions(body_data, writer);
}
```

**Step 2: Write the OpenAI completions handler**

```zig
// src/api/openai.zig
//
// OpenAI-compatible /v1/chat/completions handler.
// Phase 1: returns a mock SSE stream with fake tokens.

const std = @import("std");
const Io = std.Io;
const response = @import("../http/response.zig");

/// Mock tokens to stream in Phase 1. In later phases this will read from
/// the scheduler's completion ring.
const mock_tokens = [_][]const u8{
    "Hello", "!", " I'm", " Flint", ",", " an", " inference", " server", ".", "",
};

pub fn handleChatCompletions(
    body_data: []const u8,
    writer: *Io.Writer,
) !void {
    // Basic validation: body must be non-empty JSON
    _ = body_data; // TODO Phase 1: we don't parse the request body yet

    try response.writeSseHeaders(writer);

    // Stream mock tokens
    var buf: [1024]u8 = undefined;
    for (mock_tokens, 0..) |token, i| {
        if (token.len == 0) continue; // skip empty
        const chunk = formatChunk(&buf, token, i == mock_tokens.len - 2) catch continue;
        try response.writeSseEvent(writer, &buf, chunk);
    }

    try response.writeSseDone(writer);
}

/// Format a single OpenAI-compatible SSE chunk.
fn formatChunk(buf: []u8, token: []const u8, is_last: bool) ![]const u8 {
    const finish_reason = if (is_last) "\"stop\"" else "null";
    return std.fmt.bufPrint(buf,
        \\{{"id":"chatcmpl-mock","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":{s}}}]}}
    , .{ token, finish_reason }) catch return error.Overflow;
}
```

**Step 3: Commit**

```bash
git add src/api/router.zig src/api/openai.zig
git commit -m "feat: OpenAI-compatible API router with mock SSE streaming"
```

---

### Task 6: TCP Server + Connection Handler

**Files:**
- Create: `src/net/server.zig`
- Create: `src/net/connection.zig`

**Step 1: Write the accept loop**

Follow pike's listener.zig pattern exactly.

```zig
// src/net/server.zig

const std = @import("std");
const net = std.Io.net;
const Io = std.Io;
const connection = @import("connection.zig");
const log = std.log.scoped(.server);

pub fn runServer(gpa: std.mem.Allocator, io: Io, port: u16) !void {
    const listen_addr: net.IpAddress = .{
        .ip4 = .{ .bytes = .{ 0, 0, 0, 0 }, .port = port },
    };
    var server = try listen_addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    log.info("flint listening on port {d}", .{port});

    while (true) {
        const client = server.accept(io) catch |err| {
            log.err("accept failed: {}", .{err});
            continue;
        };
        _ = io.async(connection.handleConnection, .{ gpa, io, client });
    }
}
```

**Step 2: Write the connection handler**

Adapted from pike's connection.zig — but instead of forwarding to a backend, we parse the request and route to our API handlers.

```zig
// src/net/connection.zig
//
// Per-connection handler. Each connection is a fiber that reads HTTP requests,
// routes them to API handlers, and writes responses. Supports keep-alive.

const std = @import("std");
const net = std.Io.net;
const Io = std.Io;
const Parser = @import("http_parser").Parser;
const router = @import("../api/router.zig");
const response = @import("../http/response.zig");
const log = std.log.scoped(.connection);

pub fn handleConnection(gpa: std.mem.Allocator, io: Io, client: net.Stream) void {
    defer client.close(io);

    var read_buf: [8192]u8 = undefined;
    var write_buf: [8192]u8 = undefined;
    var reader = client.reader(io, &read_buf);
    var writer = client.writer(io, &write_buf);

    var parser = Parser.init(gpa);
    defer parser.deinit();

    while (true) {
        // Read and parse until we have a complete message
        while (!parser.message_complete) {
            const data = reader.peekGreedy(1) catch |err| switch (err) {
                error.EndOfStream => return,
                else => {
                    log.debug("read error: {}", .{err});
                    return;
                },
            };

            _ = parser.feed(data) catch |err| {
                log.debug("parse error: {}", .{err});
                response.writeResponse(
                    &writer.interface,
                    .bad_request,
                    "text/plain",
                    "Bad Request",
                ) catch return;
                return;
            };

            reader.toss(data.len);
        }

        // Route the parsed request
        const req = parser.request orelse {
            log.debug("no parsed request after message_complete", .{});
            return;
        };

        log.debug("{s} {s}", .{ @tagName(req.method), req.url });

        router.route(req, parser.body(), &writer.interface) catch |err| {
            log.debug("handler error: {}", .{err});
            response.writeResponse(
                &writer.interface,
                .internal_server_error,
                "text/plain",
                "Internal Server Error",
            ) catch return;
            return;
        };

        // Keep-alive: reset parser and loop for next request
        if (!req.keep_alive) break;
        parser.reset();
    }
}
```

**Step 3: Commit**

```bash
git add src/net/server.zig src/net/connection.zig
git commit -m "feat: TCP accept loop and connection handler with HTTP routing"
```

---

### Task 7: Wire Everything Together — Entry Point

**Files:**
- Modify: `src/main.zig`

**Step 1: Update main.zig**

```zig
const std = @import("std");
const server = @import("net/server.zig");

// Test discovery — pull in all modules
test {
    _ = @import("http_parser");
    _ = @import("net/connection.zig");
    _ = @import("api/router.zig");
    _ = @import("api/openai.zig");
    _ = @import("http/response.zig");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    // Parse optional port argument
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next(); // skip program name
    const port: u16 = if (args.next()) |port_str|
        std.fmt.parseInt(u16, port_str, 10) catch 8080
    else
        8080;

    try server.runServer(gpa, io, port);
}
```

**Step 2: Run all tests**

```bash
zig build test
```

Expected: All parser + types tests pass.

**Step 3: Build and test manually**

```bash
zig build run
```

In another terminal:
```bash
# Health check
curl -s http://localhost:8080/health
# Expected: {"status":"ok"}

# Models endpoint
curl -s http://localhost:8080/v1/models
# Expected: {"object":"list","data":[{"id":"mock-model",...}]}

# Mock SSE stream
curl -sN -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}'
# Expected: SSE events with mock tokens, ending with [DONE]

# 404 for unknown path
curl -s http://localhost:8080/unknown
# Expected: Not Found
```

**Step 4: Commit**

```bash
git add src/main.zig
git commit -m "feat: wire up entry point — flint serves HTTP + mock SSE"
```

---

### Task 8: Integration Test

**Files:**
- Create: `tests/integration/test_http_server.zig`
- Modify: `build.zig` (add integration test step)

**Step 1: Write integration test**

Test the full stack by connecting as a TCP client. Uses `std.testing.io` (Io.Threaded backend available in tests).

```zig
// tests/integration/test_http_server.zig
//
// Integration test: starts the server, sends HTTP requests via TCP, verifies responses.

const std = @import("std");
const net = std.Io.net;
const Io = std.Io;
const connection = @import("connection");

test "GET /health returns 200 OK JSON" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    // Create a socket pair (or use loopback)
    const server_addr: net.IpAddress = .{
        .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 }, // port 0 = ephemeral
    };
    var server = try server_addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    // Get the actual port assigned
    // Spawn connection handler for the client we'll connect
    const connect_addr = server.listen_address;

    // Spawn a task to accept one connection and handle it
    const accept_task = io.async(acceptOne, .{ allocator, io, &server });

    // Connect as client
    var client = try connect_addr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    var read_buf: [8192]u8 = undefined;
    var write_buf: [8192]u8 = undefined;
    var reader = client.reader(io, &read_buf);
    var writer = client.writer(io, &write_buf);

    // Send GET /health
    try writer.writeAll("GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    try writer.flush();

    // Read response
    const resp_data = try reader.allocRemaining(allocator, 8192);
    defer allocator.free(resp_data);

    // Verify response contains expected status and body
    try std.testing.expect(std.mem.startsWith(u8, resp_data, "HTTP/1.1 200 OK"));
    try std.testing.expect(std.mem.indexOf(u8, resp_data, "{\"status\":\"ok\"}") != null);

    accept_task.await(io);
}

fn acceptOne(allocator: std.mem.Allocator, io: Io, server: *net.Server) void {
    const client = server.accept(io) catch return;
    connection.handleConnection(allocator, io, client);
}
```

**Step 2: Add integration test to build.zig**

Add after the existing test step in `build.zig`:

```zig
    // Connection module for integration tests
    const connection_module = b.createModule(.{
        .root_source_file = b.path("src/net/connection.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "http_parser", .module = http_parser_module },
        },
    });

    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration/test_http_server.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "connection", .module = connection_module },
            },
        }),
    });
    const run_integration = b.addRunArtifact(integration_tests);
    test_step.dependOn(&run_integration.step);

    const integration_step = b.step("test-integration", "Run integration tests");
    integration_step.dependOn(&run_integration.step);
```

**Step 3: Create test directory and run**

```bash
mkdir -p tests/integration
zig build test
```

Expected: All tests pass (unit + integration).

**Step 4: Commit**

```bash
git add tests/ build.zig
git commit -m "feat: integration test for HTTP server"
```

---

## Summary of Tasks

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1 | Project scaffolding | `build.zig`, `build.zig.zon`, `src/main.zig` | `zig build` compiles |
| 2 | HTTP types | `src/http/types.zig` | Method/Header/Request unit tests |
| 3 | HTTP parser (llhttp) | `src/http/parser.zig` | GET, POST+body, split feeds, errors, keep-alive |
| 4 | Response writer + SSE | `src/http/response.zig` | Tested via integration |
| 5 | API router + OpenAI handler | `src/api/router.zig`, `src/api/openai.zig` | Tested via integration |
| 6 | TCP server + connection | `src/net/server.zig`, `src/net/connection.zig` | Tested via integration |
| 7 | Wire together (main.zig) | `src/main.zig` | Manual curl tests |
| 8 | Integration test | `tests/integration/test_http_server.zig` | Automated TCP-level test |

## Notes

- **All std.Io APIs are from pike** — proven working on Zig 0.16. Do NOT use APIs from the CLAUDE.md code sketches if they differ from pike.
- **llhttp hash** is from pike's `build.zig.zon` — known good.
- **Connection handler** uses pike's `peekGreedy(1)` + `toss(n)` pattern for incremental parsing.
- **No TLS** in Phase 1 — defer to later phase. Can reuse pike's TlsStream when needed.
- **fingerprint** in `build.zig.zon` — if `zig build` complains about the fingerprint, use whatever value it suggests.
- **`std.testing.io`** — this is how pike's tests get an `Io` instance. It provides a threaded backend suitable for testing.
- **`server.listen_address`** — check pike's test files to confirm the exact field name for getting the ephemeral port. May be different.
- **Module imports across directories** — `src/net/connection.zig` importing `src/api/router.zig` uses relative path `../api/router.zig`. The `http_parser` import uses the named module from `build.zig`. This matches pike's pattern.
