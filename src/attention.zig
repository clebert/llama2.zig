const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Checkpoint = @import("checkpoint.zig");
const Matrix = @import("./matrix.zig");
const VectorArray = @import("./vector_array.zig").VectorArray([]f32);

allocator: std.mem.Allocator,

n_heads: usize,
n_query_groups: usize,
head_size: usize,
head_size_sqrt: f32,
sequence_length: usize,

query_projection_matrices: []const Matrix,
key_projection_matrices: []const Matrix,
value_projection_matrices: []const Matrix,
output_projection_matrices: []const Matrix,

input_vector: []f32,
output_vector: []f32,

query_vectors: VectorArray,
key_cache: []f32,
value_cache: []f32,
scores: []f32,

pub fn init(
    allocator: std.mem.Allocator,
    checkpoint: *const Checkpoint,
    sequence_length: usize,
) !Self {
    const embedding_size = checkpoint.embedding_size;
    const n_layers = checkpoint.n_layers;
    const n_heads = checkpoint.n_heads;
    const n_query_groups = checkpoint.n_query_groups;
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

    const key_value_cache_size = n_layers * sequence_length * n_query_groups * head_size;
    const key_cache = try allocator.alloc(f32, key_value_cache_size);

    errdefer allocator.free(key_cache);

    const value_cache = try allocator.alloc(f32, key_value_cache_size);

    errdefer allocator.free(value_cache);

    const scores = try allocator.alloc(f32, sequence_length);

    errdefer allocator.free(scores);

    return Self{
        .allocator = allocator,

        .n_heads = n_heads,
        .n_query_groups = n_query_groups,
        .head_size = head_size,
        .head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(head_size))),
        .sequence_length = sequence_length,

        .query_projection_matrices = weights.attention_query_projection_matrices,
        .key_projection_matrices = weights.attention_key_projection_matrices,
        .value_projection_matrices = weights.attention_value_projection_matrices,
        .output_projection_matrices = weights.attention_output_projection_matrices,

        .input_vector = input_vector,
        .output_vector = output_vector,

        .query_vectors = query_vectors,
        .key_cache = key_cache,
        .value_cache = value_cache,
        .scores = scores,
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.input_vector);
    self.allocator.free(self.output_vector);
    self.allocator.free(self.query_vectors.data);
    self.allocator.free(self.key_cache);
    self.allocator.free(self.value_cache);
    self.allocator.free(self.scores);
}

pub fn forward(self: *const Self, layer: usize, position: usize) !void {
    const query_projection_matrix = self.query_projection_matrices[layer];
    const key_projection_matrix = self.key_projection_matrices[layer];
    const value_projection_matrix = self.value_projection_matrices[layer];
    const output_projection_matrix = self.output_projection_matrices[layer];

    const multi_head_query = self.query_vectors.data;
    const multi_head_key = self.getCacheSlice(.key, layer, position, null);
    const multi_head_value = self.getCacheSlice(.value, layer, position, null);

    query_projection_matrix.multiplyVector(self.input_vector, multi_head_query);
    key_projection_matrix.multiplyVector(self.input_vector, multi_head_key);
    value_projection_matrix.multiplyVector(self.input_vector, multi_head_value);

    self.applyRotaryPositionEmbedding(position, multi_head_key);

    for (0..self.n_heads) |head| {
        self.computeGroupedQueryAttention(layer, position, head);
    }

    output_projection_matrix.multiplyVector(self.input_vector, self.output_vector);
}

// https://arxiv.org/abs/2104.09864
fn applyRotaryPositionEmbedding(
    self: *const Self,
    position: usize,
    multi_head_key: []f32,
) void {
    @setFloatMode(.Optimized);

    const multi_head_query = self.query_vectors.data;

    std.debug.assert(multi_head_query.len % multi_head_key.len == 0);

    var index: usize = 0;

    while (index < multi_head_query.len) : (index += 2) {
        const head: f32 = @floatFromInt(index % self.head_size);

        const frequency =
            1 / std.math.pow(f32, 10000, head / @as(f32, @floatFromInt(self.head_size)));

        const rotation_scaling_factor: f32 = @as(f32, @floatFromInt(position)) * frequency;
        const real_rotation_value: f32 = std.math.cos(rotation_scaling_factor);
        const imag_rotation_value: f32 = std.math.sin(rotation_scaling_factor);

        const q_0 = multi_head_query[index];
        const q_1 = multi_head_query[index + 1];

        multi_head_query[index] = q_0 * real_rotation_value - q_1 * imag_rotation_value;
        multi_head_query[index + 1] = q_0 * imag_rotation_value + q_1 * real_rotation_value;

        if (index < multi_head_key.len) {
            const k_0 = multi_head_key[index];
            const k_1 = multi_head_key[index + 1];

            multi_head_key[index] = k_0 * real_rotation_value - k_1 * imag_rotation_value;
            multi_head_key[index + 1] = k_0 * imag_rotation_value + k_1 * real_rotation_value;
        }
    }
}

// https://arxiv.org/abs/1706.03762
fn computeGroupedQueryAttention(
    self: *const Self,
    layer: usize,
    current_position: usize,
    head: usize,
) void {
    @setFloatMode(.Optimized);

    const query_group = head / (self.n_heads / self.n_query_groups);
    const query_vector = self.query_vectors.at(head);
    const next_position = current_position + 1;

    for (0..next_position) |position| {
        const key_vector = self.getCacheSlice(.key, layer, position, query_group);

        self.scores[position] = lib.dot(query_vector, key_vector) / self.head_size_sqrt;
    }

    lib.softmax(self.scores[0..next_position]);

    const attention_values = VectorArray.init(self.head_size, self.input_vector).at(head);

    @memset(attention_values, 0);

    for (0..next_position) |position| {
        const value_vector = self.getCacheSlice(.value, layer, position, query_group);

        const weight = self.scores[position];

        for (0..self.head_size) |index| {
            attention_values[index] += value_vector[index] * weight;
        }
    }
}

const CacheType = enum { key, value };

fn getCacheSlice(
    self: *const Self,
    cache_type: CacheType,
    layer: usize,
    position: usize,
    query_group: ?usize,
) []f32 {
    const cache = if (cache_type == .key) self.key_cache else self.value_cache;
    const multi_head_cache_size = self.n_query_groups * self.head_size;

    const layer_cache_size = self.sequence_length * multi_head_cache_size;
    const layer_cache_offset = layer * layer_cache_size;
    const layer_cache = cache[layer_cache_offset..][0..layer_cache_size];

    const multi_head_cache_offset = position * multi_head_cache_size;
    const multi_head_cache = layer_cache[multi_head_cache_offset..][0..multi_head_cache_size];

    if (query_group) |group| {
        return multi_head_cache[(group * self.head_size)..][0..self.head_size];
    }

    return multi_head_cache;
}
