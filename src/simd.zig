const std = @import("std");

// Pre-normalization using RMSNorm: https://arxiv.org/abs/1910.07467
pub fn computeRMSNorm(
    comptime TValue: type,
    comptime vector_size: comptime_int,
    input_values: []const TValue,
    weight_values: []const TValue,
    output_values: []TValue,
) void {
    @setFloatMode(.Optimized);

    var rms_scaling_factor = computeScalarProduct(TValue, vector_size, input_values, input_values);

    rms_scaling_factor /= @floatFromInt(input_values.len);
    rms_scaling_factor += 1e-5;
    rms_scaling_factor = 1 / std.math.sqrt(rms_scaling_factor);

    computeVectorMultiplication(
        TValue,
        vector_size,
        rms_scaling_factor,
        input_values,
        weight_values,
        output_values,
    );
}

pub fn computeScalarProduct(
    comptime TValue: type,
    comptime vector_size: comptime_int,
    values_1: []const TValue,
    values_2: []const TValue,
) f32 {
    @setFloatMode(.Optimized);

    std.debug.assert(values_1.len == values_2.len);
    std.debug.assert(values_1.len % vector_size == 0);

    var output_values: @Vector(vector_size, f32) = @splat(0.0);
    var index: usize = 0;

    while (index < values_1.len) : (index += vector_size) {
        output_values +=
            @as(@Vector(vector_size, f32), values_1[index..][0..vector_size].*) *
            @as(@Vector(vector_size, f32), values_2[index..][0..vector_size].*);
    }

    return @reduce(.Add, output_values);
}

pub fn computeVectorAddition(
    comptime TValue: type,
    comptime vector_size: comptime_int,
    input_values_1: []const TValue,
    input_values_2: []const TValue,
    output_values: []TValue,
) void {
    @setFloatMode(.Optimized);

    std.debug.assert(input_values_1.len == input_values_2.len);
    std.debug.assert(input_values_1.len % vector_size == 0);

    var index: usize = 0;

    while (index < input_values_1.len) : (index += vector_size) {
        output_values[index..][0..vector_size].* =
            @as(@Vector(vector_size, TValue), input_values_1[index..][0..vector_size].*) +
            @as(@Vector(vector_size, TValue), input_values_2[index..][0..vector_size].*);
    }
}

pub fn computeVectorMultiplication(
    comptime TValue: type,
    comptime vector_size: comptime_int,
    scaling_factor: f32,
    input_values_1: []const TValue,
    input_values_2: []const TValue,
    output_values: []TValue,
) void {
    @setFloatMode(.Optimized);

    std.debug.assert(input_values_1.len == input_values_2.len);
    std.debug.assert(input_values_1.len == output_values.len);
    std.debug.assert(input_values_1.len % vector_size == 0);

    const scaling_factors: @Vector(vector_size, f32) = @splat(scaling_factor);

    var index: usize = 0;

    while (index < input_values_1.len) : (index += vector_size) {
        output_values[index..][0..vector_size].* =
            scaling_factors *
            @as(@Vector(vector_size, TValue), input_values_1[index..][0..vector_size].*) *
            @as(@Vector(vector_size, TValue), input_values_2[index..][0..vector_size].*);
    }
}
