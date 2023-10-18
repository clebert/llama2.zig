const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const ConverterArgs = @import("converter_args.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var args = try ConverterArgs.init(allocator);

    defer args.deinit();

    const checkpoint = try Checkpoint.init(allocator, args.model_path);

    defer checkpoint.deinit();

    try checkpoint.writeV1(allocator, args.model_path);
}
