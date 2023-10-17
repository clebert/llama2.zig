const std = @import("std");
const Chat = @import("chat.zig");
const ChatArgs = @import("chat_args.zig");

pub fn main() !void {
    var args = try ChatArgs.init(std.heap.page_allocator);

    defer args.deinit();

    var chat = try Chat.init(std.heap.page_allocator, &args);

    defer chat.deinit();

    try chat.start(std.heap.page_allocator);
}
