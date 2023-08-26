const std = @import("std");
const Cli = @import("cli.zig");
const Generator = @import("generator.zig");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    var cli = try Cli.init(std.heap.page_allocator);

    defer cli.deinit();

    var generator = try Generator.init(std.heap.page_allocator, &cli);

    defer generator.deinit();

    try generator.generate(stdout);
}

test {
    std.testing.refAllDecls(@This());
}
