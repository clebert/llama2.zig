const std = @import("std");

// Pre-normalization using RMSNorm: https://arxiv.org/abs/1910.07467
pub fn computeRMSNorm(
    input_values: []const f32,
    weight_values: []const f32,
    output_values: []f32,
) !void {
    @setFloatMode(.Optimized);

    var scaling_factor = try computeScalarProduct(input_values, input_values);

    scaling_factor /= @floatFromInt(input_values.len);
    scaling_factor += 1e-5;
    scaling_factor = 1 / std.math.sqrt(scaling_factor);

    try computeVectorMultiplication(scaling_factor, input_values, weight_values, output_values);
}

pub fn computeScalarProduct(input_values_1: []const f32, input_values_2: []const f32) !f32 {
    @setFloatMode(.Optimized);

    std.debug.assert(input_values_1.len == input_values_2.len);

    comptime var vector_len = std.atomic.cache_line / @sizeOf(f32);

    inline while (vector_len >= 4) : (vector_len /= 2) {
        if (input_values_1.len % vector_len == 0) {
            var output_values: @Vector(vector_len, f32) = @splat(0);
            var index: usize = 0;

            while (index < input_values_1.len) : (index += vector_len) {
                output_values +=
                    @as(@Vector(vector_len, f32), input_values_1[index..][0..vector_len].*) *
                    @as(@Vector(vector_len, f32), input_values_2[index..][0..vector_len].*);
            }

            return @reduce(.Add, output_values);
        }
    }

    return error.UnsupportedVectorSize;
}

pub fn computeVectorAddition(
    input_values_1: []const f32,
    input_values_2: []const f32,
    output_values: []f32,
) !void {
    @setFloatMode(.Optimized);

    std.debug.assert(input_values_1.len == input_values_2.len);
    std.debug.assert(input_values_1.len == output_values.len);

    comptime var vector_len = std.atomic.cache_line / @sizeOf(f32);

    inline while (vector_len >= 4) : (vector_len /= 2) {
        if (input_values_1.len % vector_len == 0) {
            var index: usize = 0;

            while (index < input_values_1.len) : (index += vector_len) {
                output_values[index..][0..vector_len].* =
                    @as(@Vector(vector_len, f32), input_values_1[index..][0..vector_len].*) +
                    @as(@Vector(vector_len, f32), input_values_2[index..][0..vector_len].*);
            }

            return;
        }
    }

    return error.UnsupportedVectorSize;
}

pub fn computeVectorMultiplication(
    scaling_factor: f32,
    input_values_1: []const f32,
    input_values_2: []const f32,
    output_values: []f32,
) !void {
    @setFloatMode(.Optimized);

    std.debug.assert(input_values_1.len == input_values_2.len);
    std.debug.assert(input_values_1.len == output_values.len);

    comptime var vector_len = std.atomic.cache_line / @sizeOf(f32);

    inline while (vector_len >= 4) : (vector_len /= 2) {
        if (input_values_1.len % vector_len == 0) {
            const scaling_factors: @Vector(vector_len, f32) = @splat(scaling_factor);

            var index: usize = 0;

            while (index < input_values_1.len) : (index += vector_len) {
                output_values[index..][0..vector_len].* =
                    scaling_factors *
                    @as(@Vector(vector_len, f32), input_values_1[index..][0..vector_len].*) *
                    @as(@Vector(vector_len, f32), input_values_2[index..][0..vector_len].*);
            }

            return;
        }
    }

    return error.UnsupportedVectorSize;
}
