const std = @import("std");
const Generator = @import("generator.zig");
const GeneratorArgs = @import("generator_args.zig");

pub fn main() !void {
    var args = try GeneratorArgs.init(std.heap.page_allocator);

    defer args.deinit();

    var generator = try Generator.init(std.heap.page_allocator, &args);

    defer generator.deinit();

    try generator.generate(std.io.getStdOut().writer());
}

test {
    std.testing.refAllDecls(@This());
}
