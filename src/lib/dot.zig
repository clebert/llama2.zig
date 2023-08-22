const std = @import("std");

const max_vector_len: comptime_int = 16;
const min_vector_len: comptime_int = 4;

pub fn dot(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.Optimized);

    std.debug.assert(a.len == b.len);

    const rest_len = a.len % max_vector_len;

    std.debug.assert(rest_len % min_vector_len == 0);

    var max_len_accu: @Vector(max_vector_len, f32) = @splat(0.0);
    var index: usize = 0;

    while (index < a.len - rest_len) : (index += max_vector_len) {
        max_len_accu +=
            @as(@Vector(max_vector_len, f32), a[index..][0..max_vector_len].*) *
            @as(@Vector(max_vector_len, f32), b[index..][0..max_vector_len].*);
    }

    var result = @reduce(.Add, max_len_accu);

    if (rest_len > 0) {
        var min_len_accu: @Vector(min_vector_len, f32) = @splat(0.0);

        index = a.len - rest_len;

        while (index < a.len) : (index += min_vector_len) {
            min_len_accu +=
                @as(@Vector(min_vector_len, f32), a[index..][0..min_vector_len].*) *
                @as(@Vector(min_vector_len, f32), b[index..][0..min_vector_len].*);
        }

        result += @reduce(.Add, min_len_accu);
    }

    return result;
}
