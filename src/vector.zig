const std = @import("std");

pub fn slice(
    comptime T: type,
    allocator: std.mem.Allocator,
    vector_size: usize,
    data: T,
) ![]T {
    std.debug.assert(data.len % vector_size == 0);

    var vectors = try allocator.alloc(T, data.len / vector_size);

    for (vectors, 0..) |*vector, vector_index| {
        vector.* = data[(vector_index * vector_size)..][0..vector_size];
    }

    return vectors;
}

pub fn add(vector_a: []f32, vector_b: []const f32) void {
    @setFloatMode(.Optimized);

    std.debug.assert(vector_a.len == vector_b.len);

    for (vector_a, 0..) |*element, index| {
        element.* += vector_b[index];
    }
}

pub fn argmax(vector: []f32) usize {
    var max_index: usize = 0;
    var max_element: f32 = vector[max_index];

    for (1..vector.len) |index| {
        const element = vector[index];

        if (element > max_element) {
            max_index = index;
            max_element = element;
        }
    }

    return max_index;
}

pub fn dot(vector_a: []const f32, vector_b: []const f32) f32 {
    @setFloatMode(.Optimized);

    // const native_vector_size: usize = comptime @max(std.simd.suggestVectorSize(f32) orelse 4, 4);
    const native_vector_size: usize = 4; // TODO: the above code does not run on GitHub CI

    std.debug.assert(vector_a.len == vector_b.len);
    std.debug.assert(vector_a.len % native_vector_size == 0);

    var result: f32 = 0;
    var offset: usize = 0;

    comptime var vector_size = native_vector_size * native_vector_size;

    inline while (vector_size >= native_vector_size) : (vector_size /= native_vector_size) {
        var vector: @Vector(vector_size, f32) = @splat(0.0);
        var rest = (vector_a.len - offset) % vector_size;

        while (offset < vector_a.len - rest) : (offset += vector_size) {
            vector +=
                @as(@Vector(vector_size, f32), vector_a[offset..][0..vector_size].*) *
                @as(@Vector(vector_size, f32), vector_b[offset..][0..vector_size].*);
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

pub fn softmax(vector: []f32) void {
    @setFloatMode(.Optimized);

    var max_element: f32 = std.mem.max(f32, vector);
    var sum: f32 = 0;

    for (vector) |*element| {
        element.* = std.math.exp(element.* - max_element);
        sum += element.*;
    }

    for (vector) |*element| {
        element.* /= sum;
    }
}
