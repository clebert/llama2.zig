const Self = @This();

const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const Tensor = @import("./tensor.zig").Tensor;
const vector = @import("./vector.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
head_size: usize,
head_size_sqrt: f32,
input_buffer: Tensor(2),
output_buffer: Tensor(1),
query_buffer: Tensor(2),
key_cache: Tensor(4),
value_cache: Tensor(4),
scores: []f32,

pub fn init(allocator: std.mem.Allocator, checkpoint: Checkpoint, sequence_length: usize) !Self {
    const embedding_size = checkpoint.embedding_size;
    const n_attention_heads = checkpoint.n_attention_heads;
    const head_size: usize = embedding_size / n_attention_heads;
    const input_buffer = try Tensor(2).init(allocator, [_]usize{ n_attention_heads, head_size });

    errdefer input_buffer.deinit();

    const output_buffer = try Tensor(1).init(allocator, [_]usize{embedding_size});

    errdefer output_buffer.deinit();

    const query_buffer = try Tensor(2).init(allocator, [_]usize{ n_attention_heads, head_size });

    errdefer query_buffer.deinit();

    const n_layers = checkpoint.n_layers;
    const n_attention_query_groups = checkpoint.n_attention_query_groups;

    const key_cache = try Tensor(4).init(
        allocator,
        [_]usize{ n_layers, sequence_length, n_attention_query_groups, head_size },
    );

    errdefer key_cache.deinit();

    const value_cache = try Tensor(4).init(
        allocator,
        [_]usize{ n_layers, sequence_length, n_attention_query_groups, head_size },
    );

    errdefer value_cache.deinit();

    const scores = try allocator.alloc(f32, sequence_length);

    errdefer allocator.free(scores);

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .head_size = head_size,
        .head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(head_size))),
        .input_buffer = input_buffer,
        .output_buffer = output_buffer,
        .query_buffer = query_buffer,
        .key_cache = key_cache,
        .value_cache = value_cache,
        .scores = scores,
    };
}

pub fn deinit(self: *const Self) void {
    self.input_buffer.deinit();
    self.output_buffer.deinit();
    self.query_buffer.deinit();
    self.key_cache.deinit();
    self.value_cache.deinit();
    self.allocator.free(self.scores);
}

pub fn forward(self: *const Self, layer: usize, position: usize) !void {
    const weights = self.checkpoint.weights;
    const query_matrix = weights.attention_query_matrices.slice(layer);
    const key_matrix = weights.attention_key_matrices.slice(layer);
    const value_matrix = weights.attention_value_matrices.slice(layer);
    const output_matrix = weights.attention_output_matrices.slice(layer);
    const key_buffer = self.key_cache.slice(layer).slice(position);
    const value_buffer = self.value_cache.slice(layer).slice(position);

    query_matrix.multiplyVector(self.input_buffer, self.query_buffer);
    key_matrix.multiplyVector(self.input_buffer, key_buffer);
    value_matrix.multiplyVector(self.input_buffer, value_buffer);

    self.rope(position, key_buffer);

    for (0..self.checkpoint.n_attention_heads) |head| {
        self.gqa(layer, position, head);
    }

    output_matrix.multiplyVector(self.input_buffer, self.output_buffer);
}

// Rotary positional embeddings: https://arxiv.org/abs/2104.09864
fn rope(self: *const Self, position: usize, key_buffer: Tensor(2)) void {
    @setFloatMode(.Optimized);

    std.debug.assert(self.query_buffer.data.len % key_buffer.data.len == 0);

    var index: usize = 0;

    while (index < self.query_buffer.data.len) : (index += 2) {
        const head: f32 = @floatFromInt(index % self.head_size);

        const frequency =
            1 / std.math.pow(f32, 10000, head / @as(f32, @floatFromInt(self.head_size)));

        const rotation_scaling_factor: f32 = @as(f32, @floatFromInt(position)) * frequency;
        const real_rotation_value: f32 = std.math.cos(rotation_scaling_factor);
        const imag_rotation_value: f32 = std.math.sin(rotation_scaling_factor);

        const q_0 = self.query_buffer.data[index];
        const q_1 = self.query_buffer.data[index + 1];

        self.query_buffer.data[index] = q_0 * real_rotation_value - q_1 * imag_rotation_value;
        self.query_buffer.data[index + 1] = q_0 * imag_rotation_value + q_1 * real_rotation_value;

        if (index < key_buffer.data.len) {
            const k_0 = key_buffer.data[index];
            const k_1 = key_buffer.data[index + 1];

            key_buffer.data[index] = k_0 * real_rotation_value - k_1 * imag_rotation_value;
            key_buffer.data[index + 1] = k_0 * imag_rotation_value + k_1 * real_rotation_value;
        }
    }
}

// Grouped-query attention: https://arxiv.org/abs/2305.13245v1
fn gqa(self: *const Self, layer: usize, current_position: usize, head: usize) void {
    @setFloatMode(.Optimized);

    const query_vector = self.query_buffer.slice(head);

    const query_group =
        head / (self.checkpoint.n_attention_heads / self.checkpoint.n_attention_query_groups);

    const next_position = current_position + 1;

    for (0..next_position) |position| {
        const key_vector = self.key_cache.slice(layer).slice(position).slice(query_group);

        self.scores[position] =
            vector.dot(query_vector.data, key_vector.data) / self.head_size_sqrt;
    }

    vector.softmax(self.scores[0..next_position]);

    const attention_buffer = self.input_buffer.slice(head);

    @memset(attention_buffer.data, 0);

    for (0..next_position) |position| {
        const value_vector = self.value_cache.slice(layer).slice(position).slice(query_group);
        const weight = self.scores[position];

        for (0..self.head_size) |index| {
            attention_buffer.data[index] += value_vector.data[index] * weight;
        }
    }
}
