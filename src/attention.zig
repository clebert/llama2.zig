const Self = @This();

const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const math = @import("math.zig");
const simd = @import("simd.zig");
const Vector = @import("vector.zig");

checkpoint: Checkpoint,
head_size: usize,
head_size_sqrt: f32,
input: Vector,
output: Vector,
multi_query: Vector,
key_cache: []const []const Vector,
value_cache: []const []const Vector,
scores: []f32,

pub fn createLeaky(
    allocator: std.mem.Allocator,
    checkpoint: Checkpoint,
    sequence_length: usize,
) !Self {
    const head_size = checkpoint.embedding_size / checkpoint.n_attention_heads;
    const key_cache = try allocator.alloc([]Vector, checkpoint.n_layers);

    for (key_cache) |*layer| {
        layer.* = try Vector.createMultipleLeaky(
            allocator,
            sequence_length,
            checkpoint.n_attention_query_groups * head_size,
        );
    }

    const value_cache = try allocator.alloc([]Vector, checkpoint.n_layers);

    for (value_cache) |*layer| {
        layer.* = try Vector.createMultipleLeaky(
            allocator,
            sequence_length,
            checkpoint.n_attention_query_groups * head_size,
        );
    }

    return .{
        .checkpoint = checkpoint,
        .head_size = head_size,
        .head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(head_size))),
        .input = try Vector.createLeaky(allocator, checkpoint.embedding_size),
        .output = try Vector.createLeaky(allocator, checkpoint.embedding_size),
        .multi_query = try Vector.createLeaky(allocator, checkpoint.embedding_size),
        .key_cache = key_cache,
        .value_cache = value_cache,
        .scores = try allocator.alloc(f32, sequence_length),
    };
}

pub fn forward(self: Self, layer: usize, position: usize) !void {
    const query_weight = self.checkpoint.attention_query_weights[layer];
    const key_weight = self.checkpoint.attention_key_weights[layer];
    const value_weight = self.checkpoint.attention_value_weights[layer];
    const output_weight = self.checkpoint.attention_output_weights[layer];
    const multi_key = self.key_cache[layer][position];
    const multi_value = self.value_cache[layer][position];

    try query_weight.multiplyVector(self.input, self.multi_query);
    try key_weight.multiplyVector(self.input, multi_key);
    try value_weight.multiplyVector(self.input, multi_value);

    self.computeRoPE(position, multi_key.data);

    for (0..self.checkpoint.n_attention_heads) |head| {
        try self.computeGQA(layer, position, head);
    }

    try output_weight.multiplyVector(self.input, self.output);
}

// Rotary positional embeddings: https://arxiv.org/abs/2104.09864
fn computeRoPE(self: Self, position: usize, multi_key_data: []f32) void {
    @setFloatMode(.Optimized);

    const multi_query_data = self.multi_query.data;

    std.debug.assert(multi_query_data.len % multi_key_data.len == 0);

    var index: usize = 0;

    while (index < multi_query_data.len) : (index += 2) {
        const head: f32 = @floatFromInt(index % self.head_size);

        const frequency =
            1 / std.math.pow(f32, 10000, head / @as(f32, @floatFromInt(self.head_size)));

        const rotation_scaling_factor: f32 = @as(f32, @floatFromInt(position)) * frequency;
        const real_rotation: f32 = std.math.cos(rotation_scaling_factor);
        const imag_rotation: f32 = std.math.sin(rotation_scaling_factor);

        const q_0 = multi_query_data[index];
        const q_1 = multi_query_data[index + 1];

        multi_query_data[index] = q_0 * real_rotation - q_1 * imag_rotation;
        multi_query_data[index + 1] = q_0 * imag_rotation + q_1 * real_rotation;

        if (index < multi_key_data.len) {
            const k_0 = multi_key_data[index];
            const k_1 = multi_key_data[index + 1];

            multi_key_data[index] = k_0 * real_rotation - k_1 * imag_rotation;
            multi_key_data[index + 1] = k_0 * imag_rotation + k_1 * real_rotation;
        }
    }
}

// Grouped-query attention: https://arxiv.org/abs/2305.13245v1
fn computeGQA(self: Self, layer: usize, current_position: usize, head: usize) !void {
    @setFloatMode(.Optimized);

    const query_data = self.multi_query.data[head * self.head_size ..][0..self.head_size];

    const query_group =
        head / (self.checkpoint.n_attention_heads / self.checkpoint.n_attention_query_groups);

    const next_position = current_position + 1;

    for (0..next_position) |position| {
        const multi_key = self.key_cache[layer][position];
        const key_data = multi_key.data[query_group * self.head_size ..][0..self.head_size];

        self.scores[position] =
            try simd.computeScalarProduct(query_data, key_data) / self.head_size_sqrt;
    }

    math.softmax(self.scores[0..next_position]);

    const attention_data = self.input.data[head * self.head_size ..][0..self.head_size];

    @memset(attention_data, 0);

    for (0..next_position) |position| {
        const multi_value = self.value_cache[layer][position];
        const value_data = multi_value.data[query_group * self.head_size ..][0..self.head_size];
        const weight = self.scores[position];

        for (0..self.head_size) |index| {
            attention_data[index] += value_data[index] * weight;
        }
    }
}
