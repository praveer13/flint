//! HTTP/1.1 parser — Zig-idiomatic wrapper around llhttp.
//!
//! This module will expose a zero-copy request parser that operates on
//! the connection's recv buffer, returning slices into that buffer for
//! method, path, headers, and body. For now it's a placeholder that
//! proves the llhttp C dependency compiles and links.

const types = @import("types.zig");

test {
    _ = types;
}
