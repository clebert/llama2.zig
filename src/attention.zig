const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Checkpoint = @import("checkpoint.zig");
const KeyValueCache = @import("key_value_cache.zig");
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
scores: []f32,
key_cache: KeyValueCache,
value_cache: KeyValueCache,

pub fn init(allocator: std.mem.Allocator, checkpoint: Checkpoint, sequence_length: usize) !Self {
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

    const scores = try allocator.alloc(f32, sequence_length);

    errdefer allocator.free(scores);

    const key_cache = try KeyValueCache.init(
        allocator,
        n_layers,
        n_groups,
        head_size,
        sequence_length,
    );

    errdefer key_cache.deinit();

    const value_cache = try KeyValueCache.init(
        allocator,
        n_layers,
        n_groups,
        head_size,
        sequence_length,
    );

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
        .key_cache = key_cache,
        .value_cache = value_cache,
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.input_vector);
    self.allocator.free(self.output_vector);
    self.allocator.free(self.scores);
    self.allocator.free(self.query_vectors.data);
    self.key_cache.deinit();
    self.value_cache.deinit();
}

pub fn forward(self: *const Self, position: usize, layer: usize) !void {
    const query_data = self.query_vectors.data;
    const key_data = self.key_cache.at(layer, null).at(position);
    const value_data = self.value_cache.at(layer, null).at(position);

    try self.query_projection_matrices.multiplyVector(layer, self.input_vector, query_data);
    try self.key_projection_matrices.multiplyVector(layer, self.input_vector, key_data);
    try self.value_projection_matrices.multiplyVector(layer, self.input_vector, value_data);

    lib.rope(position, self.head_size, query_data, key_data);

    for (0..self.n_heads) |head| {
        self.compute_attention(position, layer, head);
    }

    try self.output_projection_matrices.multiplyVector(
        layer,
        self.input_vector,
        self.output_vector,
    );
}

fn compute_attention(self: *const Self, current_position: usize, layer: usize, head: usize) void {
    @setFloatMode(.Optimized);

    const group = head / (self.n_heads / self.n_groups);
    const query_vector = self.query_vectors.at(head);
    const next_position = current_position + 1;

    for (0..next_position) |position| {
        const key_vector = self.key_cache.at(layer, position).at(group);

        self.scores[position] = lib.dot(query_vector, key_vector) / self.head_size_sqrt;
    }

    lib.softmax(self.scores[0..next_position]);

    const attention_values = VectorArray.init(self.head_size, self.input_vector).at(head);

    @memset(attention_values, 0);

    for (0..next_position) |position| {
        const value_vector = self.value_cache.at(layer, position).at(group);
        const weight = self.scores[position];

        for (0..self.head_size) |index| {
            attention_values[index] += value_vector[index] * weight;
        }
    }
}
