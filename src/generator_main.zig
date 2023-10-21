const std = @import("std");
const Generator = @import("generator.zig");
const GeneratorArgs = @import("generator_args.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const args = try GeneratorArgs.createLeaky(arena.allocator());

    var generator = try Generator.createLeaky(arena.allocator(), args);

    try generator.generate(std.io.getStdOut().writer());
}

test {
    std.testing.refAllDecls(@This());
}
