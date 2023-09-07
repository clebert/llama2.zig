const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Checkpoint = @import("checkpoint.zig");
const MatrixArray = @import("./matrix_array.zig");
const VectorArray = @import("./vector_array.zig").VectorArray([]f32);

allocator: std.mem.Allocator,

n_heads: usize,
n_groups: usize,
head_size: usize,
head_size_sqrt: f32,
sequence_length: usize,

query_projection_matrices: MatrixArray,
key_projection_matrices: MatrixArray,
value_projection_matrices: MatrixArray,
output_projection_matrices: MatrixArray,

input_vector: []f32,
output_vector: []f32,

query_vectors: VectorArray,
key_cache_data: []f32,
value_cache_data: []f32,
scores: []f32,

pub fn init(
    allocator: std.mem.Allocator,
    checkpoint: *const Checkpoint,
    sequence_length: usize,
) !Self {
    const embedding_size = checkpoint.embedding_size;
    const n_layers = checkpoint.n_layers;
    const n_heads = checkpoint.n_heads;
    const n_groups = checkpoint.n_groups;
    const weights = checkpoint.weights;

    const input_vector = try allocator.alloc(f32, embedding_size);

    errdefer allocator.free(input_vector);

    const output_vector = try allocator.alloc(f32, embedding_size);

    errdefer allocator.free(output_vector);

    const head_size: usize = embedding_size / n_heads;

    const query_vectors = VectorArray.init(
        head_size,
        try allocator.alloc(f32, n_heads * head_size),
    );

    errdefer allocator.free(query_vectors.data);

    const key_value_cache_data_size = n_layers * sequence_length * n_groups * head_size;
    const key_cache_data = try allocator.alloc(f32, key_value_cache_data_size);

    errdefer allocator.free(key_cache_data);

    const value_cache_data = try allocator.alloc(f32, key_value_cache_data_size);

    errdefer allocator.free(value_cache_data);

    const scores = try allocator.alloc(f32, sequence_length);

    errdefer allocator.free(scores);

    return Self{
        .allocator = allocator,

        .n_heads = n_heads,
        .n_groups = n_groups,
        .head_size = head_size,
        .head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(head_size))),
        .sequence_length = sequence_length,

        .query_projection_matrices = weights.attention_query_matrices,
        .key_projection_matrices = weights.attention_key_matrices,
        .value_projection_matrices = weights.attention_value_matrices,
        .output_projection_matrices = weights.attention_output_matrices,

        .input_vector = input_vector,
        .output_vector = output_vector,

        .scores = scores,
        .query_vectors = query_vectors,
        .key_cache_data = key_cache_data,
        .value_cache_data = value_cache_data,
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.input_vector);
    self.allocator.free(self.output_vector);
    self.allocator.free(self.query_vectors.data);
    self.allocator.free(self.key_cache_data);
    self.allocator.free(self.value_cache_data);
    self.allocator.free(self.scores);
}

pub fn forward(self: *const Self, layer: usize, position: usize) !void {
    const query_data = self.query_vectors.data;
    const key_data = self.getCacheDataSlice(self.key_cache_data, layer, position, null);
    const value_data = self.getCacheDataSlice(self.value_cache_data, layer, position, null);

    try self.query_projection_matrices.multiplyVector(layer, self.input_vector, query_data);
    try self.key_projection_matrices.multiplyVector(layer, self.input_vector, key_data);
    try self.value_projection_matrices.multiplyVector(layer, self.input_vector, value_data);

    self.applyRotaryPositionEmbedding(position, key_data);

    for (0..self.n_heads) |head| {
        self.computeAttention(position, layer, head);
    }

    try self.output_projection_matrices.multiplyVector(
        layer,
        self.input_vector,
        self.output_vector,
    );
}

// RoFormer: Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/abs/2104.09864)
fn applyRotaryPositionEmbedding(self: *const Self, position: usize, key_data: []f32) void {
    @setFloatMode(.Optimized);

    const query_data = self.query_vectors.data;

    std.debug.assert(key_data.len <= query_data.len);

    var index: usize = 0;

    while (index < query_data.len) : (index += 2) {
        const head: f32 = @floatFromInt(index % self.head_size);

        const frequency =
            1 / std.math.pow(f32, 10000, head / @as(f32, @floatFromInt(self.head_size)));

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

fn computeAttention(self: *const Self, current_position: usize, layer: usize, head: usize) void {
    @setFloatMode(.Optimized);

    const group = head / (self.n_heads / self.n_groups);
    const query_vector = self.query_vectors.at(head);
    const next_position = current_position + 1;

    for (0..next_position) |position| {
        const key_vector = self.getCacheDataSlice(self.key_cache_data, layer, position, group);

        self.scores[position] = lib.dot(query_vector, key_vector) / self.head_size_sqrt;
    }

    lib.softmax(self.scores[0..next_position]);

    const attention_values = VectorArray.init(self.head_size, self.input_vector).at(head);

    @memset(attention_values, 0);

    for (0..next_position) |position| {
        const value_vector = self.getCacheDataSlice(self.value_cache_data, layer, position, group);
        const weight = self.scores[position];

        for (0..self.head_size) |index| {
            attention_values[index] += value_vector[index] * weight;
        }
    }
}

fn getCacheDataSlice(
    self: *const Self,
    cache_data: []f32,
    layer: usize,
    position: usize,
    optional_group: ?usize,
) []f32 {
    const sequence_cache_size = self.n_groups * self.head_size;

    const layer_cache_size = self.sequence_length * sequence_cache_size;
    const layer_cache_offset = layer * layer_cache_size;
    const layer_cache_data = cache_data[layer_cache_offset..][0..layer_cache_size];

    const position_cache_offset = position * sequence_cache_size;
    const position_cache_data = layer_cache_data[position_cache_offset..][0..sequence_cache_size];

    if (optional_group) |group| {
        return position_cache_data[(group * self.head_size)..][0..self.head_size];
    }

    return position_cache_data;
}
