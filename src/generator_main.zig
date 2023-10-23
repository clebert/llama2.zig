const std = @import("std");
const Generator = @import("generator.zig");
const GeneratorArgs = @import("generator_args.zig");
const Worker = @import("worker.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const args = try GeneratorArgs.initLeaky(arena.allocator());
    const workers = try arena.allocator().alloc(Worker, args.worker_count);

    for (workers) |*worker| {
        worker.* = .{};

        try worker.spawn();
    }

    var generator = try Generator.initLeaky(arena.allocator(), args);

    try generator.generate(std.io.getStdOut().writer(), workers);
}

test {
    std.testing.refAllDecls(@This());
}
