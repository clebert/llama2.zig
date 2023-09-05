const std = @import("std");

// Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
pub fn rmsnorm(input: []const f32, weight: []const f32, output: []f32) void {
    @setFloatMode(.Optimized);

    std.debug.assert(output.len == input.len);
    std.debug.assert(output.len == weight.len);

    var rms_scaling_factor: f32 = 0;

    for (input) |element| {
        rms_scaling_factor += element * element;
    }

    rms_scaling_factor /= @floatFromInt(input.len);
    rms_scaling_factor += 1e-5;
    rms_scaling_factor = 1 / std.math.sqrt(rms_scaling_factor);

    for (output, 0..) |*element, index| {
        element.* = weight[index] * rms_scaling_factor * input[index];
    }
}
