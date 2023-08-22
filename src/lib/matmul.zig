const std = @import("std");

const dot = @import("dot.zig").dot;

pub fn matmul(result: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(b.len == result.len * a.len);

    for (result, 0..) |*entry, i| {
        entry.* = dot(a, b[(i * a.len)..][0..a.len]);
    }
}

pub fn matmul2(args_1: anytype, args_2: anytype, multi_threaded: bool) !void {
    const cpu_count = std.Thread.getCpuCount() catch 1;

    if (multi_threaded and cpu_count > 2) {
        const thread_1 = try std.Thread.spawn(.{}, matmul, args_1);
        const thread_2 = try std.Thread.spawn(.{}, matmul, args_2);

        thread_1.join();
        thread_2.join();
    } else {
        @call(.auto, matmul, args_1);
        @call(.auto, matmul, args_2);
    }
}

pub fn matmul3(args_1: anytype, args_2: anytype, args_3: anytype, multi_threaded: bool) !void {
    const cpu_count = std.Thread.getCpuCount() catch 1;

    if (multi_threaded and cpu_count > 3) {
        const thread_1 = try std.Thread.spawn(.{}, matmul, args_1);
        const thread_2 = try std.Thread.spawn(.{}, matmul, args_2);
        const thread_3 = try std.Thread.spawn(.{}, matmul, args_3);

        thread_1.join();
        thread_2.join();
        thread_3.join();
    } else {
        @call(.auto, matmul, args_1);
        @call(.auto, matmul, args_2);
        @call(.auto, matmul, args_3);
    }
}
