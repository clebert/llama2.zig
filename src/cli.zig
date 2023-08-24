const Self = @This();
const std = @import("std");

checkpoint_path: []const u8,
temperature: f32,
top_p: f32,
random_seed: u64,
n_steps: usize,
input_prompt: []const u8,
tokenizer_path: []const u8,
mmap: bool,
test_mode: bool,

arg_iterator: std.process.ArgIterator,

const Option = enum { temperature, top_p, random_seed, n_steps, input_prompt, tokenizer_path };

pub fn init(self: *Self, allocator: std.mem.Allocator) !void {
    self.arg_iterator = try std.process.argsWithAllocator(allocator);

    var current_option: ?Option = null;
    var temperature: ?f32 = null;
    var top_p: ?f32 = null;
    var random_seed: ?u64 = null;
    var n_steps: ?usize = null;
    var input_prompt: ?[]const u8 = null;
    var tokenizer_path: ?[]const u8 = null;
    var mmap: bool = true;
    var test_mode: bool = false;

    _ = self.arg_iterator.next().?;

    const checkpoint_path = self.arg_iterator.next() orelse try exit();

    while (self.arg_iterator.next()) |arg| {
        if (current_option) |option| {
            if (option == .temperature and temperature == null) {
                temperature = try std.fmt.parseFloat(f32, arg);
            } else if (option == .top_p and top_p == null) {
                top_p = try std.fmt.parseFloat(f32, arg);
            } else if (option == .random_seed and random_seed == null) {
                random_seed = try std.fmt.parseInt(u64, arg, 10);
            } else if (option == .n_steps and n_steps == null) {
                n_steps = try std.fmt.parseInt(usize, arg, 10);
            } else if (option == .input_prompt and input_prompt == null) {
                input_prompt = arg;
            } else if (option == .tokenizer_path and tokenizer_path == null) {
                tokenizer_path = arg;
            } else {
                try exit();
            }

            current_option = null;
        } else if (std.mem.eql(u8, arg, "-t")) {
            current_option = .temperature;
        } else if (std.mem.eql(u8, arg, "-p")) {
            current_option = .top_p;
        } else if (std.mem.eql(u8, arg, "-s")) {
            current_option = .random_seed;
        } else if (std.mem.eql(u8, arg, "-n")) {
            current_option = .n_steps;
        } else if (std.mem.eql(u8, arg, "-i")) {
            current_option = .input_prompt;
        } else if (std.mem.eql(u8, arg, "-z")) {
            current_option = .tokenizer_path;
        } else if (std.mem.eql(u8, arg, "--no-mmap") and mmap) {
            mmap = false;
        } else if (std.mem.eql(u8, arg, "--test") and !test_mode) {
            test_mode = true;
        } else {
            try exit();
        }
    }

    if (current_option != null) {
        try exit();
    }

    self.checkpoint_path = checkpoint_path;
    self.temperature = @max(@min(temperature orelse 1, 1), 0);
    self.top_p = @max(@min(top_p orelse 0.9, 1), 0);
    self.random_seed = random_seed orelse @intCast(std.time.milliTimestamp());
    self.n_steps = @max(n_steps orelse 256, 1);
    self.input_prompt = input_prompt orelse "";
    self.tokenizer_path = tokenizer_path orelse "tokenizer.bin";
    self.mmap = mmap;
    self.test_mode = test_mode;
}

pub fn deinit(self: *Self) void {
    self.arg_iterator.deinit();
}

fn exit() !noreturn {
    const stderr = std.io.getStdErr().writer();

    try stderr.print("Usage: llama2 <checkpoint_path> [options]\n\n", .{});

    try stderr.print("Options:\n", .{});
    try stderr.print("  -t <float>  temperature    = 1\n", .{});
    try stderr.print("  -p <float>  top_p          = 0.9; 1 == off\n", .{});
    try stderr.print("  -s <int>    random_seed    = milli_timestamp\n", .{});
    try stderr.print("  -n <int>    n_steps        = 256\n", .{});
    try stderr.print("  -i <string> input_prompt   = \"\"\n", .{});
    try stderr.print("  -z <string> tokenizer_path = \"tokenizer.bin\"\n", .{});
    try stderr.print("  --no-mmap\n\n", .{});

    try stderr.print("Example: llama2 model.bin -i \"Once upon a time\"\n", .{});

    std.process.exit(1);
}
