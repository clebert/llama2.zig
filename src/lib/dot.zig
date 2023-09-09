const std = @import("std");

pub fn dot(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.Optimized);

    const native_vector_size: usize = comptime @max(std.simd.suggestVectorSize(f32) orelse 4, 4);

    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len % native_vector_size == 0);

    var result: f32 = 0;
    var offset: usize = 0;

    comptime var vector_size = native_vector_size * native_vector_size;

    inline while (vector_size >= native_vector_size) : (vector_size /= native_vector_size) {
        var vector: @Vector(vector_size, f32) = @splat(0.0);
        var rest = (a.len - offset) % vector_size;

        while (offset < a.len - rest) : (offset += vector_size) {
            vector +=
                @as(@Vector(vector_size, f32), a[offset..][0..vector_size].*) *
                @as(@Vector(vector_size, f32), b[offset..][0..vector_size].*);
        }

        result += @reduce(.Add, vector);
    }

    return result;
}
