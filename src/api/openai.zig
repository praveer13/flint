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
//! ## Dual mode: mock (Phase 1) vs scheduler-backed (Phase 3+)
//!
//! When `scheduler` is null, the handler streams a fixed sequence of mock
//! tokens — useful for integration tests and development without a GPU.
//! When `scheduler` is non-null, the handler submits the request to the
//! scheduler, polls for tokens via `SequenceStatus`, and streams them as
//! SSE events. In Phase 3, token IDs are formatted as `[token:<id>]`
//! since no tokenizer is available yet; real detokenization comes in Phase 4.

const std = @import("std");
const Io = std.Io;

const response = @import("../http/response.zig");
const Scheduler = @import("../scheduler/scheduler.zig").Scheduler;
const PendingRequest = @import("../scheduler/scheduler.zig").PendingRequest;

/// Mock tokens streamed when no scheduler is available (Phase 1 fallback).
///
/// These form the sentence "Hello! I'm Flint, an inference server." and
/// exist purely to exercise the SSE streaming path end-to-end.
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

/// Handles a `/v1/chat/completions` request.
///
/// `body_data` is the raw JSON request body. `writer` is the connection's
/// buffered writer for sending the response. `scheduler` is optional:
/// when non-null, the request is submitted to the scheduler and real
/// tokens are streamed; when null, mock tokens are streamed (Phase 1).
pub fn handleChatCompletions(body_data: []const u8, writer: *Io.Writer, scheduler: ?*Scheduler) !void {
    if (scheduler) |sched| {
        return handleScheduledCompletions(body_data, writer, sched);
    } else {
        return handleMockCompletions(writer);
    }
}

/// Stream tokens from the scheduler (Phase 3+).
///
/// Submits a request to the scheduler, then polls the SequenceStatus
/// for new tokens. Each token is formatted as `[token:<id>]` since we
/// don't have a tokenizer yet. The loop exits when the scheduler marks
/// the sequence as done (EOS or max_tokens).
fn handleScheduledCompletions(body_data: []const u8, writer: *Io.Writer, scheduler: *Scheduler) !void {
    _ = body_data; // Phase 3: body parsing is not yet implemented.

    // Submit a request with default parameters. In later phases, these
    // will be parsed from the JSON body.
    const req = PendingRequest{
        .num_prompt_tokens = 32,
        .max_tokens = 10,
        .temperature = @as(f16, 1.0),
        .top_p = @as(f16, 1.0),
        .priority = 0,
    };

    const seq_slot = scheduler.submitRequest(req) orelse {
        // Scheduler queue is full — return 429 Too Many Requests.
        try response.writeResponse(writer, .too_many_requests, "application/json",
            \\{"error":{"message":"Server is overloaded, please retry later","type":"rate_limit_error"}}
        );
        return;
    };

    // Begin the SSE stream.
    try response.writeSseHeaders(writer);

    // Poll for tokens from the scheduler.
    var buf: [512]u8 = undefined;
    var event_buf: [1024]u8 = undefined;
    var last_count: u32 = 0;

    while (true) {
        const status = scheduler.getStatus(seq_slot);
        const current = @atomicLoad(u32, &status.tokens_generated, .acquire);

        if (current > last_count) {
            // New token(s) arrived.
            const token_id = @atomicLoad(u32, &status.last_token_id, .acquire);
            const is_done_val = @atomicLoad(u8, &status.is_done, .acquire);
            last_count = current;

            if (is_done_val == 1) {
                // Write the token chunk, then the final stop chunk, then [DONE].
                const chunk = std.fmt.bufPrint(&buf,
                    \\{{"id":"chatcmpl-flint","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"[token:{d}]"}},"finish_reason":null}}]}}
                , .{token_id}) catch return error.Overflow;
                try response.writeSseEvent(writer, &event_buf, chunk);

                const stop_chunk = std.fmt.bufPrint(&buf,
                    \\{{"id":"chatcmpl-flint","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}
                , .{}) catch return error.Overflow;
                try response.writeSseEvent(writer, &event_buf, stop_chunk);

                try response.writeSseDone(writer);
                return;
            }

            // Non-final token.
            const chunk = std.fmt.bufPrint(&buf,
                \\{{"id":"chatcmpl-flint","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"[token:{d}]"}},"finish_reason":null}}]}}
            , .{token_id}) catch return error.Overflow;
            try response.writeSseEvent(writer, &event_buf, chunk);
        }

        // Brief yield to avoid burning CPU while waiting for the next token.
        // On Io.Evented this would ideally yield the fiber; for now we use
        // a spin hint which is cheap and sufficient for Phase 3.
        std.atomic.spinLoopHint();
    }
}

/// Stream mock tokens (Phase 1 fallback when no scheduler is available).
fn handleMockCompletions(writer: *Io.Writer) !void {
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
