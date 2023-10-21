const std = @import("std");
const Chat = @import("chat.zig");
const ChatArgs = @import("chat_args.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const args = try ChatArgs.createLeaky(arena.allocator());

    var chat = try Chat.createLeaky(arena.allocator(), args);

    try chat.start(arena.allocator());
}
