//! Shared memory region — memory-mapped file handle for cross-process IPC.
//!
//! Flint uses shared memory (not sockets or pipes) for communication between
//! the Zig server process and the Python GPU worker processes. Both sides
//! memory-map the same file and read/write through raw pointers. This avoids
//! serialization overhead, system call overhead on the data path, and Python
//! GIL contention — the only synchronization is CPU cache coherence via
//! atomic loads and stores.
//!
//! The workflow:
//!   1. The Zig server calls `create()` to make a file on `/dev/shm/` (a
//!      tmpfs mount backed by RAM, not disk) and maps it into its address
//!      space.
//!   2. It writes ring buffer headers, block tables, and other structures
//!      into the mapped region using `ptrAt()` and `sliceAt()`.
//!   3. The Python worker calls the equivalent of `open()` (via mmap in
//!      Python) on the same path, getting its own virtual-address mapping
//!      of the same physical pages.
//!   4. Both sides read and write through their mappings. Writes by one
//!      process are visible to the other because `MAP_SHARED` makes the
//!      kernel share the underlying page frames.
//!
//! This module is a thin wrapper around `mmap`, `openat`, and `ftruncate`.
//! It does not use `std.Io` — shared memory regions are created once at
//! startup and then accessed via pointer arithmetic and atomics, not I/O
//! syscalls.

const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;

/// Minimum page size for this platform. All mmap regions are aligned to
/// this boundary. On x86_64 Linux this is 4096 bytes (4 KiB).
const page_size = std.heap.page_size_min;

/// A handle to a memory-mapped shared file.
///
/// The region exposes the mapped memory as a raw byte pointer. Higher-level
/// modules (ring buffers, block tables) overlay typed structs on top of this
/// byte range using `ptrAt` and `sliceAt`.
///
/// Both `create` and `open` produce a valid `ShmRegion`. The difference is
/// that `create` makes a new file (truncating any existing one) and sets its
/// size, while `open` attaches to an existing file whose size is discovered
/// via `statx`.
pub const ShmRegion = struct {
    /// Pointer to the start of the mapped region. Aligned to the system
    /// page size because `mmap` always returns page-aligned addresses.
    ptr: [*]align(page_size) u8,

    /// Total size of the mapped region in bytes.
    len: usize,

    /// File descriptor for the underlying shared memory file. Kept open
    /// so that the mapping remains valid. Closed by `close()`.
    fd: posix.fd_t,

    /// Errors that can occur when creating a new shared memory region.
    pub const CreateError = posix.OpenError || posix.MMapError || FtruncateError;

    /// Errors that can occur when opening an existing shared memory region.
    pub const OpenError = posix.OpenError || posix.MMapError || StatxError;

    /// Errors from the `ftruncate` syscall.
    pub const FtruncateError = error{
        /// The file descriptor is not open for writing, or refers to a
        /// type that cannot be truncated.
        AccessDenied,
        /// An I/O error occurred while modifying the file size.
        InputOutput,
        /// The requested size would exceed the process or system limits.
        TooBig,
        /// An unexpected errno was returned by the kernel.
        Unexpected,
    };

    /// Errors from the `statx` syscall.
    pub const StatxError = error{
        /// The file does not exist or the path is invalid.
        FileNotFound,
        /// Permission was denied.
        AccessDenied,
        /// An unexpected errno was returned by the kernel.
        Unexpected,
    };

    /// Create a new shared memory region.
    ///
    /// This creates (or truncates) the file at `path`, sets it to `size`
    /// bytes, and maps the entire file into this process's address space
    /// with read-write permissions. The mapping is `MAP_SHARED`, meaning
    /// writes are visible to any other process that maps the same file.
    ///
    /// Typical usage:
    /// ```
    /// var region = try ShmRegion.create("/dev/shm/flint.ring", 4096);
    /// defer region.close();
    /// ```
    ///
    /// `size` must be greater than zero.
    pub fn create(path: [*:0]const u8, size: usize) CreateError!ShmRegion {
        // Open (or create + truncate) the file. O_RDWR is required because
        // we need both read and write access to the mapping.
        const fd = try posix.openatZ(
            posix.AT.FDCWD,
            path,
            .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true },
            0o600,
        );
        errdefer closeFd(fd);

        // Set the file to the requested size. mmap requires the file to be
        // at least as large as the mapping; without ftruncate the file would
        // be zero bytes and mmap would fail or produce SIGBUS on access.
        try ftruncate(fd, size);

        // Map the file into our address space. PROT_READ | PROT_WRITE lets
        // us read and write through the pointer. MAP_SHARED means changes
        // are visible to other processes that map the same file (this is the
        // whole point of shared memory IPC).
        const mapped = try posix.mmap(
            null,
            size,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .SHARED },
            fd,
            0,
        );

        return .{
            .ptr = mapped.ptr,
            .len = mapped.len,
            .fd = fd,
        };
    }

    /// Open an existing shared memory region.
    ///
    /// Attaches to a file that was previously created (typically by the Zig
    /// server via `create`). The file size is discovered automatically with
    /// `statx`, and the entire file is mapped read-write.
    ///
    /// This is the path the Python worker would take (though in Python it
    /// uses `mmap.mmap()` directly — this Zig version exists for testing
    /// and for any Zig code that needs to attach to an existing region).
    pub fn open(path: [*:0]const u8) OpenError!ShmRegion {
        // Open existing file — no CREAT, no TRUNC.
        const fd = try posix.openatZ(
            posix.AT.FDCWD,
            path,
            .{ .ACCMODE = .RDWR },
            0o600,
        );
        errdefer closeFd(fd);

        // Discover the file size using statx with AT_EMPTY_PATH. We pass
        // the fd as the dirfd and an empty path, which tells the kernel to
        // stat the fd itself. We only request the SIZE field.
        const size = try fstatSize(fd);

        const mapped = try posix.mmap(
            null,
            size,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .SHARED },
            fd,
            0,
        );

        return .{
            .ptr = mapped.ptr,
            .len = mapped.len,
            .fd = fd,
        };
    }

    /// Unmap the memory and close the file descriptor.
    ///
    /// After calling `close()`, the `ptr` is invalid and must not be
    /// dereferenced. The underlying file is NOT deleted — it persists on
    /// `/dev/shm/` until explicitly unlinked (with `std.os.linux.unlink`
    /// or `rm`). This is intentional: the Zig process may close and reopen
    /// regions during restarts without disrupting workers.
    pub fn close(self: *ShmRegion) void {
        // munmap releases the virtual address range back to the kernel.
        // The physical pages remain because other processes may still have
        // the file mapped.
        posix.munmap(@alignCast(self.ptr[0..self.len]));

        // Close the file descriptor. The file itself stays on disk.
        closeFd(self.fd);

        // Poison the fields to catch use-after-close bugs in debug builds.
        self.ptr = undefined;
        self.len = 0;
        self.fd = -1;
    }

    /// Get a typed pointer at a byte offset within the region.
    ///
    /// This is how higher-level modules overlay structures on the raw byte
    /// range. For example, a ring buffer header at offset 0:
    ///
    /// ```
    /// const header = region.ptrAt(RingHeader, 0);
    /// header.tail.store(0, .release);
    /// ```
    ///
    /// The caller is responsible for ensuring:
    ///   - `offset + @sizeOf(T) <= self.len` (no out-of-bounds access)
    ///   - `offset` is properly aligned for `T`
    ///
    /// In debug/safe builds these constraints are checked at runtime.
    pub fn ptrAt(self: *const ShmRegion, comptime T: type, offset: usize) *T {
        std.debug.assert(offset + @sizeOf(T) <= self.len);
        return @ptrCast(@alignCast(self.ptr + offset));
    }

    /// Get a typed slice at a byte offset within the region.
    ///
    /// Returns a slice of `count` elements of type `T` starting at `offset`
    /// bytes from the beginning of the region. Useful for arrays in shared
    /// memory, such as ring buffer slot arrays or block table rows.
    ///
    /// ```
    /// const slots = region.sliceAt(u64, 128, 64);  // 64 u64s starting at byte 128
    /// slots[0] = 42;
    /// ```
    ///
    /// Same alignment and bounds requirements as `ptrAt`.
    pub fn sliceAt(self: *const ShmRegion, comptime T: type, offset: usize, count: usize) []T {
        std.debug.assert(offset + @sizeOf(T) * count <= self.len);
        const start: [*]T = @ptrCast(@alignCast(self.ptr + offset));
        return start[0..count];
    }

    // -- internal helpers --------------------------------------------------

    /// Wrapper around the Linux `ftruncate` syscall.
    ///
    /// Zig 0.16's `std.posix` does not expose a high-level `ftruncate`
    /// wrapper, so we call the raw Linux syscall and translate the errno.
    fn ftruncate(fd: posix.fd_t, size: usize) FtruncateError!void {
        const rc = linux.ftruncate(fd, @intCast(size));
        return switch (linux.errno(rc)) {
            .SUCCESS => {},
            .ACCES, .PERM => error.AccessDenied,
            .IO => error.InputOutput,
            .FBIG => error.TooBig,
            else => error.Unexpected,
        };
    }

    /// Get the size of an open file descriptor using the Linux `statx`
    /// syscall.
    ///
    /// Uses `AT_EMPTY_PATH` so we can stat by fd without needing the path
    /// again.
    fn fstatSize(fd: posix.fd_t) StatxError!usize {
        // AT_EMPTY_PATH = 0x1000: interpret an empty pathname relative to
        // dirfd, effectively doing fstat(fd).
        const AT_EMPTY_PATH: u32 = 0x1000;

        var stat_buf: linux.Statx = undefined;
        const rc = linux.statx(
            fd,
            "",
            AT_EMPTY_PATH,
            .{ .SIZE = true },
            &stat_buf,
        );

        return switch (linux.errno(rc)) {
            .SUCCESS => @intCast(stat_buf.size),
            .NOENT => error.FileNotFound,
            .ACCES => error.AccessDenied,
            else => error.Unexpected,
        };
    }

    /// Close a raw file descriptor via the Linux syscall.
    fn closeFd(fd: posix.fd_t) void {
        _ = linux.close(fd);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Path used by all tests. Lives on tmpfs (`/dev/shm`) so no disk I/O.
const test_path: [*:0]const u8 = "/dev/shm/flint_test_region";

/// Remove the test file if it exists. Called at the start and end of each
/// test to avoid leaking state between runs.
fn cleanupTestFile() void {
    _ = linux.unlink(test_path);
}

test "create a region, write and read a u64 at offset 0" {
    cleanupTestFile();
    defer cleanupTestFile();

    var region = try ShmRegion.create(test_path, 4096);
    defer region.close();

    // Write a known value at the start of the region.
    const ptr: *u64 = @ptrCast(@alignCast(region.ptr));
    ptr.* = 0xDEAD_BEEF_CAFE_BABE;

    // Read it back through the raw pointer.
    const val: *const u64 = @ptrCast(@alignCast(region.ptr));
    try testing.expectEqual(@as(u64, 0xDEAD_BEEF_CAFE_BABE), val.*);
}

test "ptrAt returns a correctly typed pointer" {
    cleanupTestFile();
    defer cleanupTestFile();

    var region = try ShmRegion.create(test_path, 4096);
    defer region.close();

    // Write through a typed pointer obtained from ptrAt.
    const val_ptr = region.ptrAt(u32, 64);
    val_ptr.* = 42;

    // Verify the bytes are at the expected offset.
    const raw: *const u32 = @ptrCast(@alignCast(region.ptr + 64));
    try testing.expectEqual(@as(u32, 42), raw.*);
}

test "sliceAt returns a correctly typed slice" {
    cleanupTestFile();
    defer cleanupTestFile();

    var region = try ShmRegion.create(test_path, 4096);
    defer region.close();

    // Write an array of u16 values through a slice.
    const slice = region.sliceAt(u16, 128, 8);
    for (slice, 0..) |*elem, i| {
        elem.* = @intCast(i * 10);
    }

    // Read them back and verify.
    try testing.expectEqual(@as(u16, 0), slice[0]);
    try testing.expectEqual(@as(u16, 30), slice[3]);
    try testing.expectEqual(@as(u16, 70), slice[7]);
}

test "open reads data written by create (proves shared mapping)" {
    cleanupTestFile();
    defer cleanupTestFile();

    // Phase 1: create and write.
    {
        var region = try ShmRegion.create(test_path, 4096);
        defer region.close();

        const ptr = region.ptrAt(u64, 0);
        ptr.* = 0x1234_5678_9ABC_DEF0;
    }

    // Phase 2: open and read — the data written above must be visible
    // because both mappings share the same underlying file.
    {
        var region = try ShmRegion.open(test_path);
        defer region.close();

        try testing.expectEqual(@as(usize, 4096), region.len);

        const ptr = region.ptrAt(u64, 0);
        try testing.expectEqual(@as(u64, 0x1234_5678_9ABC_DEF0), ptr.*);
    }
}

test "close cleans up without crashing; file persists until unlinked" {
    cleanupTestFile();
    defer cleanupTestFile();

    var region = try ShmRegion.create(test_path, 4096);
    region.close();

    // The file should still exist on disk even after close().
    // Verify by opening it again.
    var region2 = try ShmRegion.open(test_path);
    region2.close();
}

test "extern struct roundtrip via ptrAt" {
    cleanupTestFile();
    defer cleanupTestFile();

    // Simulates the kind of packed ABI struct used for Zig <-> Python IPC.
    const TestStruct = extern struct {
        seq_id: u64,
        token_id: u32,
        is_eos: u8,
        _pad: [3]u8 = .{ 0, 0, 0 },
    };

    var region = try ShmRegion.create(test_path, 4096);
    defer region.close();

    const entry = region.ptrAt(TestStruct, 0);
    entry.* = .{
        .seq_id = 999,
        .token_id = 42,
        .is_eos = 1,
    };

    // Re-read through a fresh pointer to prove the bytes are correct.
    const readback = region.ptrAt(TestStruct, 0);
    try testing.expectEqual(@as(u64, 999), readback.seq_id);
    try testing.expectEqual(@as(u32, 42), readback.token_id);
    try testing.expectEqual(@as(u8, 1), readback.is_eos);
}
