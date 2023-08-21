const std = @import("std");

// RoFormer: Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/abs/2104.09864)
pub fn rope(
    pos: usize,
    head_size: usize,
    queries_buffer: []f32,
    keys_buffer: []f32,
) void {
    @setFloatMode(.Optimized);

    std.debug.assert(keys_buffer.len <= queries_buffer.len);

    var index: usize = 0;

    while (index < queries_buffer.len) : (index += 2) {
        const head_index: f32 = @floatFromInt(index % head_size);

        const frequency: f32 =
            1 / std.math.pow(f32, 10000, head_index / @as(f32, @floatFromInt(head_size)));

        const rotation_scaling_factor: f32 = @as(f32, @floatFromInt(pos)) * frequency;
        const real_rotation_value: f32 = std.math.cos(rotation_scaling_factor);
        const imag_rotation_value: f32 = std.math.sin(rotation_scaling_factor);

        const query_0 = queries_buffer[index];
        const query_1 = queries_buffer[index + 1];

        queries_buffer[index] = query_0 * real_rotation_value - query_1 * imag_rotation_value;
        queries_buffer[index + 1] = query_0 * imag_rotation_value + query_1 * real_rotation_value;

        if (index < keys_buffer.len) {
            const key_0 = keys_buffer[index];
            const key_1 = keys_buffer[index + 1];

            keys_buffer[index] = key_0 * real_rotation_value - key_1 * imag_rotation_value;
            keys_buffer[index + 1] = key_0 * imag_rotation_value + key_1 * real_rotation_value;
        }
    }
}
