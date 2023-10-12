const std = @import("std");

pub fn add(input_a: []f32, input_b: []const f32) void {
    @setFloatMode(.Optimized);

    std.debug.assert(input_a.len == input_b.len);

    for (input_a, 0..) |*element, index| {
        element.* += input_b[index];
    }
}

pub fn argmax(input: []f32) usize {
    var max_index: usize = 0;
    var max_element: f32 = input[max_index];

    for (1..input.len) |index| {
        const element = input[index];

        if (element > max_element) {
            max_index = index;
            max_element = element;
        }
    }

    return max_index;
}

pub fn dot(input_a: []const f32, input_b: []const f32) f32 {
    @setFloatMode(.Optimized);

    const native_vector_size: usize = 4;

    std.debug.assert(input_a.len == input_b.len);
    std.debug.assert(input_a.len % native_vector_size == 0);

    var result: f32 = 0;
    var offset: usize = 0;

    comptime var vector_size = native_vector_size * native_vector_size;

    inline while (vector_size >= native_vector_size) : (vector_size /= native_vector_size) {
        var vector: @Vector(vector_size, f32) = @splat(0.0);
        var rest = (input_a.len - offset) % vector_size;

        while (offset < input_a.len - rest) : (offset += vector_size) {
            vector +=
                @as(@Vector(vector_size, f32), input_a[offset..][0..vector_size].*) *
                @as(@Vector(vector_size, f32), input_b[offset..][0..vector_size].*);
        }

        result += @reduce(.Add, vector);
    }

    return result;
}

// Pre-normalization using RMSNorm: https://arxiv.org/abs/1910.07467
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

pub fn softmax(input: []f32) void {
    @setFloatMode(.Optimized);

    var max_element: f32 = std.mem.max(f32, input);
    var sum: f32 = 0;

    for (input) |*element| {
        element.* = std.math.exp(element.* - max_element);
        sum += element.*;
    }

    for (input) |*element| {
        element.* /= sum;
    }
}
