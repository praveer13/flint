//! HTTP types — request and response structures shared across the
//! protocol layer.
//!
//! These are pure data types with no I/O dependency, so they can be
//! used by both the parser and the response writer without pulling in
//! networking code.

test {
    _ = @import("std");
}
