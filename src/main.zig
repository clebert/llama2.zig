const std = @import("std");
const Chat = @import("chat.zig");
const Cli = @import("cli.zig");
const Generator = @import("generator.zig");

pub fn main() !void {
    var cli = try Cli.init(std.heap.page_allocator);

    defer cli.deinit();

    if (cli.chat) {
        var chat = try Chat.init(std.heap.page_allocator, &cli);

        defer chat.deinit();

        try chat.start(std.heap.page_allocator);
    } else {
        var generator = try Generator.init(std.heap.page_allocator, &cli);

        defer generator.deinit();

        try generator.generate(std.io.getStdOut().writer());
    }
}

test {
    std.testing.refAllDecls(@This());
}
