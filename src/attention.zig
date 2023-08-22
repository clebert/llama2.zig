const std = @import("std");

const checkpoint = @import("checkpoint.zig");
const lib = @import("lib.zig");
const utils = @import("utils.zig");

pub const Attention = struct {
    const Self = @This();

    input_buffer: []f32,
    output_buffer: []f32,
    scores_buffer: []f32,
    queries_buffer: []f32,
    keys_buffer: []f32,
    values_buffer: []f32,
    key_cache: []f32,
    value_cache: []f32,

    pub fn init(self: *Self, allocator: std.mem.Allocator, config: *const checkpoint.Config) !void {
        const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

        self.input_buffer = try allocator.alloc(f32, config.dim);
        self.output_buffer = try allocator.alloc(f32, config.dim);
        self.scores_buffer = try allocator.alloc(f32, config.n_heads * config.seq_len);
        self.queries_buffer = try allocator.alloc(f32, config.dim);
        self.keys_buffer = try allocator.alloc(f32, kv_dim);
        self.values_buffer = try allocator.alloc(f32, kv_dim);
        self.key_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim);
        self.value_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim);
    }

    pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
        allocator.free(self.input_buffer);
        allocator.free(self.output_buffer);
        allocator.free(self.scores_buffer);
        allocator.free(self.queries_buffer);
        allocator.free(self.keys_buffer);
        allocator.free(self.values_buffer);
        allocator.free(self.key_cache);
        allocator.free(self.value_cache);
    }

    pub fn forward(
        self: *const Self,
        config: *const checkpoint.Config,
        weights: *const checkpoint.Weights,
        pos: usize,
        layer: usize,
    ) !void {
        const dim = config.dim;
        const n_heads = config.n_heads;
        const seq_len = config.seq_len;
        const kv_dim = self.keys_buffer.len;
        const query_weights_dim = dim * dim;
        const kv_weights_dim = dim * kv_dim;

        try lib.matmul3(
            .{
                self.queries_buffer,
                self.input_buffer,
                weights.query[(layer * query_weights_dim)..][0..query_weights_dim],
            },
            .{
                self.keys_buffer,
                self.input_buffer,
                weights.key[(layer * kv_weights_dim)..][0..kv_weights_dim],
            },
            .{
                self.values_buffer,
                self.input_buffer,
                weights.value[(layer * kv_weights_dim)..][0..kv_weights_dim],
            },
            dim >= 4096,
        );

        const head_size = dim / n_heads;

        lib.rope(pos, head_size, self.queries_buffer, self.keys_buffer);

        const kv_cache_dim = seq_len * kv_dim;
        const kv_cache_offset = layer * kv_cache_dim;

        @memcpy(
            self.key_cache[(kv_cache_offset + pos * kv_dim)..][0..self.keys_buffer.len],
            self.keys_buffer,
        );

        @memcpy(
            self.value_cache[(kv_cache_offset + pos * kv_dim)..][0..self.values_buffer.len],
            self.values_buffer,
        );

        for (0..n_heads) |query_head| {
            self.compute_attention(query_head, head_size, config, pos, kv_cache_offset, kv_dim);
        }

        lib.matmul(
            self.output_buffer,
            self.input_buffer,
            weights.attention_output[(layer * dim * dim)..][0..(dim * dim)],
        );
    }

    fn compute_attention(
        self: *const Self,
        query_head: usize,
        head_size: usize,
        config: *const checkpoint.Config,
        current_position: usize,
        kv_cache_offset: usize,
        kv_dim: usize,
    ) void {
        const n_groups = config.n_heads / config.n_kv_heads;
        const head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(head_size)));
        const query_head_offset = query_head * head_size;
        const query_head_group = query_head / n_groups;
        const key_value_head_offset = query_head_group * head_size;

        // get the query vector for this head
        const query = self.queries_buffer[query_head_offset..][0..head_size];

        // attention scores for this head
        const attention_weights = self.scores_buffer[(query_head * config.seq_len)..];

        // iterate over all timesteps, including the current one
        for (0..(current_position + 1)) |position| {
            // get the key vector for this head and at this timestep
            const key = self.key_cache[(kv_cache_offset + position * kv_dim + key_value_head_offset)..][0..head_size];

            // calculate the attention score as the dot product of q and k
            // save the score to the attention buffer
            attention_weights[position] = lib.dotProduct(query, key) / head_size_sqrt;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        utils.softmax(attention_weights[0..(current_position + 1)]);

        // weighted sum of the values, store back into intermediate_buffer
        const intermediate_buffer = self.input_buffer[query_head_offset..][0..head_size];

        @memset(intermediate_buffer, 0);

        for (0..(current_position + 1)) |position| {
            // get the value vector for this head and at this timestep
            const value = self.value_cache[(kv_cache_offset + position * kv_dim + key_value_head_offset)..];

            // get the attention weight for this timestep
            const attention_weight = attention_weights[position];

            // accumulate the weighted value into intermediate_buffer
            for (0..head_size) |i| {
                intermediate_buffer[i] += attention_weight * value[i];
            }
        }
    }
};
