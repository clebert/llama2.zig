const std = @import("std");
const Chat = @import("chat.zig");
const ChatArgs = @import("chat_args.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var args = try ChatArgs.init(allocator);

    defer args.deinit();

    var chat = try Chat.init(allocator, args);

    defer chat.deinit();

    try chat.start(allocator);
}
