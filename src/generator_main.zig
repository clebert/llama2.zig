const std = @import("std");
const Generator = @import("generator.zig");
const GeneratorArgs = @import("generator_args.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var args = try GeneratorArgs.init(allocator);

    defer args.deinit();

    var generator = try Generator.init(allocator, args);

    defer generator.deinit();

    try generator.generate(std.io.getStdOut().writer());
}

test {
    std.testing.refAllDecls(@This());
}
