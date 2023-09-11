const Self = @This();

const std = @import("std");

checkpoint_path: []const u8,
temperature: f32,
top_p: f32,
random_seed: u64,
n_steps: usize,
prompt: []const u8,
tokenizer_path: []const u8,
chat: bool,
system_prompt: []const u8,
mmap: bool,
timer: bool,
arg_iterator: std.process.ArgIterator,

const Option = enum {
    temperature,
    top_p,
    random_seed,
    n_steps,
    prompt,
    tokenizer_path,
    mode,
    system_prompt,
};

pub fn init(allocator: std.mem.Allocator) !Self {
    var current_option: ?Option = null;
    var temperature: ?f32 = null;
    var top_p: ?f32 = null;
    var random_seed: ?u64 = null;
    var n_steps: ?usize = null;
    var prompt: ?[]const u8 = null;
    var tokenizer_path: ?[]const u8 = null;
    var mode: ?[]const u8 = null;
    var system_prompt: ?[]const u8 = null;
    var mmap: bool = true;
    var timer: bool = true;

    var arg_iterator = try std.process.argsWithAllocator(allocator);

    errdefer arg_iterator.deinit();

    _ = arg_iterator.next().?;

    const checkpoint_path = arg_iterator.next() orelse try exit();

    while (arg_iterator.next()) |arg| {
        if (current_option) |option| {
            if (option == .temperature and temperature == null) {
                temperature = try std.fmt.parseFloat(f32, arg);
            } else if (option == .top_p and top_p == null) {
                top_p = try std.fmt.parseFloat(f32, arg);
            } else if (option == .random_seed and random_seed == null) {
                random_seed = try std.fmt.parseInt(u64, arg, 10);
            } else if (option == .n_steps and n_steps == null) {
                n_steps = try std.fmt.parseInt(usize, arg, 10);
            } else if (option == .prompt and prompt == null) {
                prompt = arg;
            } else if (option == .tokenizer_path and tokenizer_path == null) {
                tokenizer_path = arg;
            } else if (option == .mode and mode == null) {
                if (std.mem.eql(u8, arg, "generate") or std.mem.eql(u8, arg, "chat")) {
                    mode = arg;
                } else {
                    try exit();
                }
            } else if (option == .system_prompt and system_prompt == null) {
                system_prompt = arg;
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
            current_option = .prompt;
        } else if (std.mem.eql(u8, arg, "-z")) {
            current_option = .tokenizer_path;
        } else if (std.mem.eql(u8, arg, "-m")) {
            current_option = .mode;
        } else if (std.mem.eql(u8, arg, "-y")) {
            current_option = .system_prompt;
        } else if (std.mem.eql(u8, arg, "--no-mmap") and mmap) {
            mmap = false;
        } else if (std.mem.eql(u8, arg, "--no-timer") and timer) {
            timer = false;
        } else {
            try exit();
        }
    }

    if (current_option != null) {
        try exit();
    }

    return Self{
        .checkpoint_path = checkpoint_path,
        .temperature = @max(@min(temperature orelse 1, 1), 0),
        .top_p = @max(@min(top_p orelse 0.9, 1), 0),
        .random_seed = random_seed orelse @intCast(std.time.milliTimestamp()),
        .n_steps = n_steps orelse 0,
        .prompt = prompt orelse "",
        .tokenizer_path = tokenizer_path orelse "tokenizer.bin",
        .chat = if (mode) |arg| std.mem.eql(u8, arg, "chat") else false,
        .system_prompt = system_prompt orelse "",
        .mmap = mmap,
        .timer = timer,
        .arg_iterator = arg_iterator,
    };
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
    try stderr.print("  -n <int>    n_steps        = max_sequence_length\n", .{});
    try stderr.print("  -i <string> prompt         = \"\"\n", .{});
    try stderr.print("  -z <string> tokenizer_path = \"tokenizer.bin\"\n", .{});
    try stderr.print("  -m <string> mode           = \"generate\"; (alt. \"chat\")\n", .{});
    try stderr.print("  -y <string> system_prompt  = \"\"\n", .{});
    try stderr.print("  --no-mmap\n", .{});
    try stderr.print("  --no-timer\n\n", .{});

    try stderr.print("Example: llama2 model.bin -i \"Once upon a time\"\n", .{});

    std.process.exit(1);
}
