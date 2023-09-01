const std = @import("std");

// Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
pub fn rmsnorm(input_vector: []const f32, weight_vector: []const f32, output_vector: []f32) void {
    @setFloatMode(.Optimized);

    std.debug.assert(output_vector.len == input_vector.len);
    std.debug.assert(output_vector.len == weight_vector.len);

    var rms_scaling_factor: f32 = 0;

    for (input_vector) |element| {
        rms_scaling_factor += element * element;
    }

    rms_scaling_factor /= @floatFromInt(input_vector.len);
    rms_scaling_factor += 1e-5;
    rms_scaling_factor = 1 / std.math.sqrt(rms_scaling_factor);

    for (output_vector, 0..) |*element, index| {
        element.* = weight_vector[index] * rms_scaling_factor * input_vector[index];
    }
}
