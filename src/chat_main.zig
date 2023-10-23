const std = @import("std");
const Chat = @import("chat.zig");
const ChatArgs = @import("chat_args.zig");
const Worker = @import("worker.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const args = try ChatArgs.initLeaky(arena.allocator());
    const workers = try arena.allocator().alloc(Worker, args.worker_count);

    for (workers) |*worker| {
        worker.* = .{};

        try worker.spawn();
    }

    var chat = try Chat.initLeaky(arena.allocator(), args);

    try chat.start(arena.allocator(), workers);
}
