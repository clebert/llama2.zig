const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Checkpoint = @import("checkpoint.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
sequence_length: usize,
input_buffer: []f32,
output_buffer: []f32,
scores_buffer: []f32,
queries_buffer: []f32,
key_cache: []f32,
value_cache: []f32,

pub fn init(allocator: std.mem.Allocator, checkpoint: Checkpoint, sequence_length: usize) !Self {
    const embedding_size = checkpoint.embedding_size;
    const input_buffer = try allocator.alloc(f32, embedding_size);

    errdefer allocator.free(input_buffer);

    const output_buffer = try allocator.alloc(f32, embedding_size);

    errdefer allocator.free(output_buffer);

    const scores_buffer = try allocator.alloc(f32, checkpoint.n_query_heads * sequence_length);

    errdefer allocator.free(scores_buffer);

    const queries_buffer = try allocator.alloc(f32, embedding_size);

    errdefer allocator.free(queries_buffer);

    const key_value_size = checkpoint.n_query_head_groups * checkpoint.query_head_size;
    const key_value_cache_size = checkpoint.n_layers * sequence_length * key_value_size;
    const key_cache = try allocator.alloc(f32, key_value_cache_size);

    errdefer allocator.free(key_cache);

    const value_cache = try allocator.alloc(f32, key_value_cache_size);

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .sequence_length = sequence_length,
        .input_buffer = input_buffer,
        .output_buffer = output_buffer,
        .scores_buffer = scores_buffer,
        .queries_buffer = queries_buffer,
        .key_cache = key_cache,
        .value_cache = value_cache,
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.input_buffer);
    self.allocator.free(self.output_buffer);
    self.allocator.free(self.scores_buffer);
    self.allocator.free(self.queries_buffer);
    self.allocator.free(self.key_cache);
    self.allocator.free(self.value_cache);
}

pub fn forward(self: *const Self, pos: usize, layer: usize) !void {
    const checkpoint = self.checkpoint;
    const weights = checkpoint.weights;

    try weights.attention_query_matrices.multiplyVector(
        layer,
        self.input_buffer,
        self.queries_buffer,
    );

    const query_head_size = checkpoint.query_head_size;
    const key_value_size = checkpoint.n_query_head_groups * query_head_size;
    const key_value_cache_offset = layer * (self.sequence_length * key_value_size);

    const key_cache = self.key_cache[key_value_cache_offset..];
    const keys_buffer = key_cache[(pos * key_value_size)..][0..key_value_size];

    const value_cache = self.value_cache[key_value_cache_offset..];
    const values_buffer = value_cache[(pos * key_value_size)..][0..key_value_size];

    try weights.attention_key_matrices.multiplyVector(layer, self.input_buffer, keys_buffer);

    lib.rope(pos, query_head_size, self.queries_buffer, keys_buffer);

    try weights.attention_value_matrices.multiplyVector(layer, self.input_buffer, values_buffer);

    for (0..checkpoint.n_query_heads) |query_head| {
        self.compute_weighted_values(pos, query_head, key_cache, value_cache);
    }

    try weights.attention_output_matrices.multiplyVector(
        layer,
        self.input_buffer,
        self.output_buffer,
    );
}

fn compute_weighted_values(
    self: *const Self,
    pos: usize,
    query_head: usize,
    key_cache: []const f32,
    value_cache: []const f32,
) void {
    @setFloatMode(.Optimized);

    const checkpoint = self.checkpoint;
    const n_query_head_groups = checkpoint.n_query_head_groups;
    const query_head_group = query_head / (checkpoint.n_query_heads / n_query_head_groups);
    const query_head_size = checkpoint.query_head_size;
    const query_head_offset = query_head * query_head_size;
    const query = self.queries_buffer[query_head_offset..][0..query_head_size];
    const key_value_size = n_query_head_groups * query_head_size;
    const key_value_head_offset = query_head_group * query_head_size;
    const scores = self.scores_buffer[(query_head * self.sequence_length)..];

    for (0..(pos + 1)) |prev_pos| {
        const key_value_cache_offset = prev_pos * key_value_size + key_value_head_offset;
        const key = key_cache[key_value_cache_offset..][0..query_head_size];

        scores[prev_pos] = lib.dot(query, key) / checkpoint.query_head_size_sqrt;
    }

    lib.softmax(scores[0..(pos + 1)]);

    const weighted_values = self.input_buffer[query_head_offset..][0..query_head_size];

    @memset(weighted_values, 0);

    for (0..(pos + 1)) |prev_pos| {
        const key_value_cache_offset = prev_pos * key_value_size + key_value_head_offset;
        const value = value_cache[key_value_cache_offset..];
        const weight = scores[prev_pos];

        for (0..query_head_size) |index| {
            weighted_values[index] += weight * value[index];
        }
    }
}
