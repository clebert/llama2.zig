const std = @import("std");

pub const Args = struct {
    checkpoint_path: []const u8,
    temperature: f32,
    random_seed: u64,
    n_steps: usize,
    input_prompt: []const u8,
};

const Option = enum { temperature, random_seed, n_steps, input_prompt };

pub fn parseArgs(allocator: std.mem.Allocator) !Args {
    var args = try std.process.argsWithAllocator(allocator);
    var current_option: ?Option = null;
    var temperature: f32 = 1.0;
    var random_seed: u64 = @intCast(std.time.milliTimestamp());
    var n_steps: usize = 256;
    var input_prompt: []const u8 = "";

    _ = args.next().?;

    const checkpoint_path = args.next() orelse {
        try printHelp();
        std.process.exit(1);
    };

    while (args.next()) |arg| {
        if (current_option) |option| {
            if (option == .temperature) {
                temperature = try std.fmt.parseFloat(f32, arg);
            } else if (option == .random_seed) {
                random_seed = try std.fmt.parseInt(u64, arg, 10);
            } else if (option == .n_steps) {
                n_steps = try std.fmt.parseInt(usize, arg, 10);
            } else if (option == .input_prompt) {
                input_prompt = arg;
            }

            current_option = null;
        } else if (std.mem.eql(u8, arg, "-t")) {
            current_option = .temperature;
        } else if (std.mem.eql(u8, arg, "-s")) {
            current_option = .random_seed;
        } else if (std.mem.eql(u8, arg, "-n")) {
            current_option = .n_steps;
        } else if (std.mem.eql(u8, arg, "-i")) {
            current_option = .input_prompt;
        } else {
            try printHelp();
            std.process.exit(1);
        }
    }

    if (current_option != null) {
        try printHelp();
        std.process.exit(1);
    }

    return Args{
        .checkpoint_path = checkpoint_path,
        .temperature = temperature,
        .random_seed = random_seed,
        .n_steps = n_steps,
        .input_prompt = input_prompt,
    };
}

fn printHelp() !void {
    const stderr = std.io.getStdErr().writer();

    try stderr.print("Usage: llama2 <checkpoint_path> [options]\n\n", .{});
    try stderr.print("Options:\n", .{});
    try stderr.print("  -t <float>  temperature  = 1.0\n", .{});
    // try stderr.print("  -p <float>  top_p        = 0.9\n", .{});
    try stderr.print("  -s <int>    random_seed  = milli_timestamp\n", .{});
    try stderr.print("  -n <int>    n_steps      = 256; 0 == max_seq_len\n", .{});
    try stderr.print("  -i <string> input_prompt = \"\"\n\n", .{});
    try stderr.print("Example: llama2 model.bin -i \"Once upon a time\"\n", .{});
}
