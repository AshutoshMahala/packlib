const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- Library module (for use as a dependency) ---
    const lib_mod = b.addModule("packlib", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // --- Unit tests (per-module, via root.zig test refs) ---
    const unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    // --- Integration tests ---
    const integration_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "packlib", .module = lib_mod },
        },
    });
    const integration_tests = b.addTest(.{
        .root_module = integration_mod,
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);

    // --- Reference test vectors ---
    const vectors_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_vectors.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "packlib", .module = lib_mod },
        },
    });
    const vector_tests = b.addTest(.{
        .root_module = vectors_mod,
    });
    const run_vector_tests = b.addRunArtifact(vector_tests);

    // --- Test step ---
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_integration_tests.step);
    test_step.dependOn(&run_vector_tests.step);

    // --- Unit test only step ---
    const unit_test_step = b.step("test-unit", "Run unit tests only");
    unit_test_step.dependOn(&run_unit_tests.step);

    // --- Integration test only step ---
    const integration_test_step = b.step("test-integration", "Run integration tests only");
    integration_test_step.dependOn(&run_integration_tests.step);

    // --- Test vectors step ---
    const vectors_test_step = b.step("test-vectors", "Run reference test vector tests");
    vectors_test_step.dependOn(&run_vector_tests.step);
}
