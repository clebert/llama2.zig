const Self = @This();

const std = @import("std");

model_path: []const u8,
random_seed: u64,
sequence_length: usize,
system_prompt: []const u8,
temperature: f32,
top_p: f32,
user_prompt: []const u8,
worker_count: usize,

const Option = enum {
    random_seed,
    sequence_length,
    system_prompt,
    temperature,
    top_p,
    user_prompt,
    worker_count,
};

pub fn initLeaky(allocator: std.mem.Allocator) !Self {
    var arg_iterator = try std.process.argsWithAllocator(allocator);

    _ = arg_iterator.next().?;

    const model_path = arg_iterator.next() orelse try help(1);

    var current_option: ?Option = null;
    var random_seed: ?u64 = null;
    var sequence_length: ?usize = null;
    var system_prompt: ?[]const u8 = null;
    var temperature: ?f32 = null;
    var top_p: ?f32 = null;
    var user_prompt: ?[]const u8 = null;
    var worker_count: ?usize = null;

    while (arg_iterator.next()) |arg| {
        if (current_option) |option| {
            if (option == .random_seed and random_seed == null) {
                random_seed = try std.fmt.parseInt(u64, arg, 10);
            } else if (option == .sequence_length and sequence_length == null) {
                sequence_length = try std.fmt.parseInt(usize, arg, 10);
            } else if (option == .system_prompt and system_prompt == null) {
                system_prompt = arg;
            } else if (option == .temperature and temperature == null) {
                temperature = try std.fmt.parseFloat(f32, arg);
            } else if (option == .top_p and top_p == null) {
                top_p = try std.fmt.parseFloat(f32, arg);
            } else if (option == .user_prompt and user_prompt == null) {
                user_prompt = arg;
            } else if (option == .worker_count and worker_count == null) {
                worker_count = try std.fmt.parseInt(usize, arg, 10);
            } else {
                try help(1);
            }

            current_option = null;
        } else if (std.mem.eql(u8, arg, "--random_seed")) {
            current_option = .random_seed;
        } else if (std.mem.eql(u8, arg, "--sequence_length")) {
            current_option = .sequence_length;
        } else if (std.mem.eql(u8, arg, "--system_prompt")) {
            current_option = .system_prompt;
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            current_option = .temperature;
        } else if (std.mem.eql(u8, arg, "--top_p")) {
            current_option = .top_p;
        } else if (std.mem.eql(u8, arg, "--user_prompt")) {
            current_option = .user_prompt;
        } else if (std.mem.eql(u8, arg, "--worker_count")) {
            current_option = .worker_count;
        } else {
            try help(if (std.mem.eql(u8, arg, "--help")) 0 else 1);
        }
    }

    if (current_option != null) {
        try help(1);
    }

    return .{
        .model_path = model_path,
        .random_seed = random_seed orelse @intCast(std.time.milliTimestamp()),
        .sequence_length = sequence_length orelse 0,
        .system_prompt = system_prompt orelse "",
        .temperature = @max(@min(temperature orelse 1, 1), 0),
        .top_p = @max(@min(top_p orelse 0.9, 1), 0),
        .user_prompt = user_prompt orelse "",
        .worker_count = worker_count orelse try std.Thread.getCpuCount(),
    };
}

fn help(exit_status: u8) !noreturn {
    const console = if (exit_status == 0)
        std.io.getStdOut().writer()
    else
        std.io.getStdErr().writer();

    try console.print("Usage: llama2-chat <model_path> [options]\n\n", .{});

    try console.print("Options:\n", .{});
    try console.print("  --help\n", .{});
    try console.print("  --random_seed     <int>    = <milli_timestamp>\n", .{});
    try console.print("  --sequence_length <int>    = <max_sequence_length>\n", .{});
    try console.print("  --system_prompt   <string> = \"\"\n", .{});
    try console.print("  --temperature     <float>  = 1.0\n", .{});
    try console.print("  --top_p           <float>  = 0.9\n", .{});
    try console.print("  --user_prompt     <string> = \"\"\n", .{});
    try console.print("  --worker_count    <int>    = <cpu_count>\n", .{});

    std.process.exit(exit_status);
}
