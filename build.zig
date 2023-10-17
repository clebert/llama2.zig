const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const chat_exe = b.addExecutable(.{
        .name = "llama2-chat",
        .root_source_file = .{ .path = "src/chat_main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const generator_exe = b.addExecutable(.{
        .name = "llama2-generator",
        .root_source_file = .{ .path = "src/generator_main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const build_options = b.addOptions();

    chat_exe.addOptions("build_options", build_options);
    generator_exe.addOptions("build_options", build_options);

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(chat_exe);
    b.installArtifact(generator_exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_chat_cmd = b.addRunArtifact(chat_exe);
    const run_generator_cmd = b.addRunArtifact(generator_exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_chat_cmd.step.dependOn(b.getInstallStep());
    run_generator_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_chat_cmd.addArgs(args);
        run_generator_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_chat_step = b.step("run-chat", "Run the chat");

    run_chat_step.dependOn(&run_chat_cmd.step);

    const run_generator_step = b.step("run-generator", "Run the generator");

    run_generator_step.dependOn(&run_generator_cmd.step);

    const test_step = b.step("test", "Run unit tests");

    const generator_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/generator_main.zig" },
        .target = target,
        .optimize = optimize,
    });

    generator_tests.addOptions("build_options", build_options);

    const run_tests = b.addRunArtifact(generator_tests);

    test_step.dependOn(&run_tests.step);
}
