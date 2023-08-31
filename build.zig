const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "llama2",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const build_options = b.addOptions();

    exe.addOptions("build_options", build_options);

    const metal = b.option(bool, "metal", "Use the Metal framework") orelse false;

    build_options.addOption(bool, "metal", metal);

    if (metal) {
        exe.linkLibC();
        exe.linkLibCpp();

        exe.linkFramework("Foundation");
        exe.linkFramework("QuartzCore");
        exe.linkFramework("Metal");

        exe.addIncludePath(.{ .path = "metal-cpp" });
        exe.addIncludePath(.{ .path = "src/lib-cpp" });

        exe.addCSourceFile(.{
            .file = .{ .path = "src/lib-cpp/matvecmul_metal.cpp" },
            .flags = &.{"-std=c++17"},
        });
    }

    const accelerate = b.option(bool, "accelerate", "Use the Accelerate framework") orelse false;

    build_options.addOption(bool, "accelerate", accelerate);

    if (accelerate) {
        exe.linkLibC();
        exe.linkLibCpp();

        exe.linkFramework("Accelerate");

        exe.addIncludePath(.{ .path = "src/lib-c" });

        exe.addCSourceFile(.{
            .file = .{ .path = "src/lib-c/matvecmul_accelerate.c" },
            .flags = &.{},
        });
    }

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");

    run_step.dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run unit tests");

    const tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    tests.addOptions("build_options", build_options);

    const run_tests = b.addRunArtifact(tests);

    test_step.dependOn(&run_tests.step);
}
