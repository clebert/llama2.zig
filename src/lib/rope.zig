const std = @import("std");

// RoFormer: Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/abs/2104.09864)
pub fn rope(position: usize, head_size: usize, query_data: []f32, key_data: []f32) void {
    @setFloatMode(.Optimized);

    std.debug.assert(key_data.len <= query_data.len);

    var index: usize = 0;

    while (index < query_data.len) : (index += 2) {
        const head: f32 = @floatFromInt(index % head_size);
        const frequency = 1 / std.math.pow(f32, 10000, head / @as(f32, @floatFromInt(head_size)));
        const rotation_scaling_factor: f32 = @as(f32, @floatFromInt(position)) * frequency;
        const real_rotation_value: f32 = std.math.cos(rotation_scaling_factor);
        const imag_rotation_value: f32 = std.math.sin(rotation_scaling_factor);

        const q_0 = query_data[index];
        const q_1 = query_data[index + 1];

        query_data[index] = q_0 * real_rotation_value - q_1 * imag_rotation_value;
        query_data[index + 1] = q_0 * imag_rotation_value + q_1 * real_rotation_value;

        if (index < key_data.len) {
            const k_0 = key_data[index];
            const k_1 = key_data[index + 1];

            key_data[index] = k_0 * real_rotation_value - k_1 * imag_rotation_value;
            key_data[index + 1] = k_0 * imag_rotation_value + k_1 * real_rotation_value;
        }
    }
}
