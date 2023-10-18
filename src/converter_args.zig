const Self = @This();

const std = @import("std");

arg_iterator: std.process.ArgIterator,
model_path: []const u8,

const Option = enum { temperature, top_p, random_seed, n_steps, prompt };

pub fn init(allocator: std.mem.Allocator) !Self {
    var arg_iterator = try std.process.argsWithAllocator(allocator);

    errdefer arg_iterator.deinit();

    _ = arg_iterator.next().?;

    const model_path = arg_iterator.next() orelse try help(1);

    while (arg_iterator.next()) |arg| {
        try help(if (std.mem.eql(u8, arg, "--help")) 0 else 1);
    }

    return Self{ .arg_iterator = arg_iterator, .model_path = model_path };
}

pub fn deinit(self: *Self) void {
    self.arg_iterator.deinit();
}

fn help(exit_status: u8) !noreturn {
    const console = if (exit_status == 0)
        std.io.getStdOut().writer()
    else
        std.io.getStdErr().writer();

    try console.print("Usage: llama2-converter <model_path> [options]\n\n", .{});

    try console.print("Options:\n", .{});
    try console.print("  --help\n", .{});

    std.process.exit(exit_status);
}
