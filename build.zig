const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const simd_diag = b.option(bool, "simd-diag", "Enable SIMD width diagnostics") orelse false;

    const options = b.addOptions();
    options.addOption(bool, "simd_diag", simd_diag);

    // Shared library for Python ctypes
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "snake",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/snake.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib.root_module.addOptions("build_options", options);

    b.installArtifact(lib);

    // Unit tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/snake.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    tests.root_module.addOptions("build_options", options);

    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run unit tests").dependOn(&run_tests.step);
}
