const std = @import("std");

const max_vector_len: comptime_int = 16;
const min_vector_len: comptime_int = 4;

pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.Optimized);

    std.debug.assert(a.len == b.len);

    const rest_len = a.len % max_vector_len;

    std.debug.assert(rest_len % min_vector_len == 0);

    var buffer_1: @Vector(max_vector_len, f32) = @splat(0.0);
    var index: usize = 0;

    while (index < a.len - rest_len) : (index += max_vector_len) {
        buffer_1 +=
            @as(@Vector(max_vector_len, f32), a[index..][0..max_vector_len].*) *
            @as(@Vector(max_vector_len, f32), b[index..][0..max_vector_len].*);
    }

    var result = @reduce(.Add, buffer_1);

    if (rest_len > 0) {
        var buffer_2: @Vector(min_vector_len, f32) = @splat(0.0);

        index = a.len - rest_len;

        while (index < a.len) : (index += min_vector_len) {
            buffer_2 +=
                @as(@Vector(min_vector_len, f32), a[index..][0..min_vector_len].*) *
                @as(@Vector(min_vector_len, f32), b[index..][0..min_vector_len].*);
        }

        result += @reduce(.Add, buffer_2);
    }

    return result;
}

pub fn matmul(result: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(b.len == result.len * a.len);

    for (result, 0..) |*entry, i| {
        entry.* = dotProduct(a, b[(i * a.len)..][0..a.len]);
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
