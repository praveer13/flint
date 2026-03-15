//! OpenAI-compatible API handler for chat completions.
//!
//! This module implements the `/v1/chat/completions` endpoint, streaming
//! tokens back to the client using Server-Sent Events (SSE) in the same
//! format as the OpenAI API.
//!
//! ## OpenAI SSE streaming format
//!
//! When `stream: true` is set (the default for Flint), the server sends
//! a series of SSE frames, each containing a JSON "chunk" object:
//!
//! ```
//! data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
//!
//! data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}
//!
//! ...
//!
//! data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
//!
//! data: [DONE]
//!
//! ```
//!
//! Each `data:` line is followed by two newlines (the SSE frame delimiter).
//! The final chunk carries `"finish_reason":"stop"` with an empty delta,
//! and the stream ends with the literal `data: [DONE]` sentinel.
//!
//! ## Phase 1 behaviour
//!
//! In this phase the handler ignores the request body entirely and streams
//! a fixed sequence of mock tokens. This lets us validate the SSE framing,
//! HTTP response format, and client-side parsing without needing a model
//! or scheduler. Later phases will parse the JSON body, tokenize the
//! prompt, submit to the scheduler, and stream real model output.

const std = @import("std");
const Io = std.Io;

const response = @import("../http/response.zig");

/// Mock tokens streamed in Phase 1.
///
/// These form the sentence "Hello! I'm Flint, an inference server." and
/// exist purely to exercise the SSE streaming path end-to-end. They will
/// be replaced by real model output once the scheduler and vLLM worker
/// are wired up in Phase 4.
const mock_tokens = [_][]const u8{
    "Hello",
    "!",
    " I'm",
    " Flint",
    ",",
    " an",
    " inference",
    " server",
    ".",
};

/// Handles a `/v1/chat/completions` request by streaming mock tokens.
///
/// `body_data` is the raw JSON request body (ignored in Phase 1).
/// `writer` is the connection's buffered writer for sending the response.
///
/// The function writes SSE headers, then emits one SSE event per token
/// in the OpenAI chunk format, and finishes with the `[DONE]` sentinel.
pub fn handleChatCompletions(body_data: []const u8, writer: *Io.Writer) !void {
    _ = body_data; // Phase 1: request body is ignored.

    // Begin the SSE stream — sends HTTP 200 with event-stream headers.
    try response.writeSseHeaders(writer);

    // Stream each mock token as an OpenAI-compatible SSE chunk.
    var buf: [512]u8 = undefined;
    for (mock_tokens, 0..) |token, i| {
        const is_last = (i == mock_tokens.len - 1);

        // Build the JSON chunk payload. The last token carries
        // "finish_reason":"stop"; all others have null.
        const chunk = if (is_last)
            std.fmt.bufPrint(&buf,
                \\{{"id":"chatcmpl-mock","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":"stop"}}]}}
            , .{token}) catch return error.Overflow
        else
            std.fmt.bufPrint(&buf,
                \\{{"id":"chatcmpl-mock","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}
            , .{token}) catch return error.Overflow;

        var event_buf: [1024]u8 = undefined;
        try response.writeSseEvent(writer, &event_buf, chunk);
    }

    // Signal end of stream.
    try response.writeSseDone(writer);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "mock_tokens list is non-empty" {
    // Sanity check — if someone accidentally empties the list, this catches it.
    try std.testing.expect(mock_tokens.len > 0);
}

test "handleChatCompletions compiles with correct signature" {
    // Verify the function signature is compatible with the router's call site.
    // Actual output testing requires an Io backend (covered by integration tests).
    _ = &handleChatCompletions;
}
