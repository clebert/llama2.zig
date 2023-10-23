const std = @import("std");

// Pre-normalization using RMSNorm: https://arxiv.org/abs/1910.07467
pub fn computeRMSNorm(input_data: []const f32, weight_data: []const f32, output_data: []f32) !void {
    @setFloatMode(.Optimized);

    var scaling_factor = try computeScalarProduct(input_data, input_data);

    scaling_factor /= @floatFromInt(input_data.len);
    scaling_factor += 1e-5;
    scaling_factor = 1 / std.math.sqrt(scaling_factor);

    try computeVectorMultiplication(scaling_factor, input_data, weight_data, output_data);
}

pub fn computeScalarProduct(input_data_1: []const f32, input_data_2: []const f32) !f32 {
    @setFloatMode(.Optimized);

    std.debug.assert(input_data_1.len == input_data_2.len);

    comptime var vector_len = std.atomic.cache_line / @sizeOf(f32);

    inline while (vector_len >= 4) : (vector_len /= 2) {
        if (input_data_1.len % vector_len == 0) {
            var output_data: @Vector(vector_len, f32) = @splat(0);
            var index: usize = 0;

            while (index < input_data_1.len) : (index += vector_len) {
                output_data +=
                    @as(@Vector(vector_len, f32), input_data_1[index..][0..vector_len].*) *
                    @as(@Vector(vector_len, f32), input_data_2[index..][0..vector_len].*);
            }

            return @reduce(.Add, output_data);
        }
    }

    return error.UnsupportedVectorSize;
}

pub fn computeVectorAddition(
    input_data_1: []const f32,
    input_data_2: []const f32,
    output_data: []f32,
) !void {
    @setFloatMode(.Optimized);

    std.debug.assert(input_data_1.len == input_data_2.len);
    std.debug.assert(input_data_1.len == output_data.len);

    comptime var vector_len = std.atomic.cache_line / @sizeOf(f32);

    inline while (vector_len >= 4) : (vector_len /= 2) {
        if (input_data_1.len % vector_len == 0) {
            var index: usize = 0;

            while (index < input_data_1.len) : (index += vector_len) {
                output_data[index..][0..vector_len].* =
                    @as(@Vector(vector_len, f32), input_data_1[index..][0..vector_len].*) +
                    @as(@Vector(vector_len, f32), input_data_2[index..][0..vector_len].*);
            }

            return;
        }
    }

    return error.UnsupportedVectorSize;
}

pub fn computeVectorMultiplication(
    scaling_factor: f32,
    input_data_1: []const f32,
    input_data_2: []const f32,
    output_data: []f32,
) !void {
    @setFloatMode(.Optimized);

    std.debug.assert(input_data_1.len == input_data_2.len);
    std.debug.assert(input_data_1.len == output_data.len);

    comptime var vector_len = std.atomic.cache_line / @sizeOf(f32);

    inline while (vector_len >= 4) : (vector_len /= 2) {
        if (input_data_1.len % vector_len == 0) {
            const scaling_factors: @Vector(vector_len, f32) = @splat(scaling_factor);

            var index: usize = 0;

            while (index < input_data_1.len) : (index += vector_len) {
                output_data[index..][0..vector_len].* =
                    scaling_factors *
                    @as(@Vector(vector_len, f32), input_data_1[index..][0..vector_len].*) *
                    @as(@Vector(vector_len, f32), input_data_2[index..][0..vector_len].*);
            }

            return;
        }
    }

    return error.UnsupportedVectorSize;
}
