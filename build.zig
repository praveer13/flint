const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // llhttp C dependency — vendored from nodejs/llhttp v9.3.1.
    // We compile the generated C source directly into our module rather than
    // linking a separate static library. This keeps the build simple and lets
    // Zig's build system handle cross-compilation of the C code automatically.
    const llhttp_dep = b.dependency("llhttp", .{});

    // HTTP parser module — wraps llhttp C library with a Zig-idiomatic API.
    // Exposed as a named module so any part of the codebase can
    // `@import("http_parser")` without fragile relative paths.
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

    // Run step: `zig build run`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the flint server");
    run_step.dependOn(&run_cmd.step);

    // Unit tests — rooted at the main module so `zig build test` discovers
    // all `test` blocks transitively through imports.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_exe_tests.step);
}
