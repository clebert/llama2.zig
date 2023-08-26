const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Checkpoint = @import("checkpoint.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
seq_len: usize,
input_buffer: []f32,
output_buffer: []f32,
scores_buffer: []f32,
queries_buffer: []f32,
keys_buffer: []f32,
values_buffer: []f32,
key_cache: []f32,
value_cache: []f32,

pub fn init(allocator: std.mem.Allocator, checkpoint: Checkpoint, seq_len: usize) !Self {
    const dim = checkpoint.dim;
    const kv_dim = checkpoint.kv_dim;
    const kv_cache_dim = checkpoint.n_layers * seq_len * kv_dim;

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .seq_len = seq_len,
        .input_buffer = try allocator.alloc(f32, dim),
        .output_buffer = try allocator.alloc(f32, dim),
        .scores_buffer = try allocator.alloc(f32, checkpoint.n_heads * seq_len),
        .queries_buffer = try allocator.alloc(f32, dim),
        .keys_buffer = try allocator.alloc(f32, kv_dim),
        .values_buffer = try allocator.alloc(f32, kv_dim),
        .key_cache = try allocator.alloc(f32, kv_cache_dim),
        .value_cache = try allocator.alloc(f32, kv_cache_dim),
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.input_buffer);
    self.allocator.free(self.output_buffer);
    self.allocator.free(self.scores_buffer);
    self.allocator.free(self.queries_buffer);
    self.allocator.free(self.keys_buffer);
    self.allocator.free(self.values_buffer);
    self.allocator.free(self.key_cache);
    self.allocator.free(self.value_cache);
}

pub fn forward(self: *const Self, pos: usize, layer: usize) !void {
    const checkpoint = self.checkpoint;
    const dim = checkpoint.dim;
    const kv_dim = checkpoint.kv_dim;
    const weights = checkpoint.weights;

    const query_weights_dim = dim * dim;
    const kv_weights_dim = dim * kv_dim;

    try lib.matmul3(
        .{
            self.queries_buffer,
            self.input_buffer,
            weights.attention_query[(layer * query_weights_dim)..][0..query_weights_dim],
        },
        .{
            self.keys_buffer,
            self.input_buffer,
            weights.attention_key[(layer * kv_weights_dim)..][0..kv_weights_dim],
        },
        .{
            self.values_buffer,
            self.input_buffer,
            weights.attention_value[(layer * kv_weights_dim)..][0..kv_weights_dim],
        },
        dim >= 4096,
    );

    lib.rope(pos, checkpoint.head_size, self.queries_buffer, self.keys_buffer);

    const kv_cache_dim = self.seq_len * kv_dim;
    const kv_cache_layer_offset = layer * kv_cache_dim;

    @memcpy(
        self.key_cache[(kv_cache_layer_offset + pos * kv_dim)..][0..self.keys_buffer.len],
        self.keys_buffer,
    );

    @memcpy(
        self.value_cache[(kv_cache_layer_offset + pos * kv_dim)..][0..self.values_buffer.len],
        self.values_buffer,
    );

    for (0..checkpoint.n_heads) |head| {
        self.compute_weighted_values(pos, head, kv_cache_layer_offset);
    }

    lib.matmul(
        self.output_buffer,
        self.input_buffer,
        weights.attention_output[(layer * dim * dim)..][0..(dim * dim)],
    );
}

fn compute_weighted_values(
    self: *const Self,
    pos: usize,
    head: usize,
    kv_cache_layer_offset: usize,
) void {
    @setFloatMode(.Optimized);

    const checkpoint = self.checkpoint;
    const kv_dim = checkpoint.kv_dim;
    const head_size = checkpoint.head_size;

    const group = head / checkpoint.n_groups;
    const kv_head_offset = group * head_size;
    const head_offset = head * head_size;
    const query = self.queries_buffer[head_offset..][0..head_size];
    const scores = self.scores_buffer[(head * self.seq_len)..];

    for (0..(pos + 1)) |prev_pos| {
        const kv_cache_head_offset = kv_cache_layer_offset + prev_pos * kv_dim + kv_head_offset;
        const key = self.key_cache[kv_cache_head_offset..][0..head_size];

        scores[prev_pos] = lib.dot(query, key) / checkpoint.head_size_sqrt;
    }

    lib.softmax(scores[0..(pos + 1)]);

    const weighted_values = self.input_buffer[head_offset..][0..head_size];

    @memset(weighted_values, 0);

    for (0..(pos + 1)) |prev_pos| {
        const kv_cache_head_offset = kv_cache_layer_offset + prev_pos * kv_dim + kv_head_offset;
        const value = self.value_cache[kv_cache_head_offset..];
        const weight = scores[prev_pos];

        for (0..head_size) |index| {
            weighted_values[index] += weight * value[index];
        }
    }
}
