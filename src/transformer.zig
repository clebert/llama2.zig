const std = @import("std");

const Attention = @import("attention.zig").Attention;
const checkpoint = @import("checkpoint.zig");
const FeedForward = @import("feed_forward.zig").FeedForward;
const lib = @import("lib.zig");

pub const Transformer = struct {
    const Self = @This();

    hidden_state: []f32,
    logits: []f32,
    attention: Attention,
    feed_forward: FeedForward,

    pub fn init(self: *Self, allocator: std.mem.Allocator, config: *const checkpoint.Config) !void {
        self.hidden_state = try allocator.alloc(f32, config.dim);
        self.logits = try allocator.alloc(f32, config.vocab_size);

        try self.attention.init(allocator, config);
        try self.feed_forward.init(allocator, config);
    }

    pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
        allocator.free(self.hidden_state);
        allocator.free(self.logits);

        self.attention.deinit(allocator);
        self.feed_forward.deinit(allocator);
    }

    pub fn forward(
        self: *const Self,
        token: usize,
        pos: usize,
        config: *const checkpoint.Config,
        weights: *const checkpoint.Weights,
    ) !void {
        @memcpy(
            self.hidden_state,
            weights.token_embedding[(token * config.dim)..][0..self.hidden_state.len],
        );

        for (0..config.n_layers) |layer| {
            lib.rmsnorm(
                self.attention.input_buffer,
                self.hidden_state,
                weights.rms_attention_input[(layer * config.dim)..][0..config.dim],
            );

            try self.attention.forward(config, weights, pos, layer);

            lib.add(self.hidden_state, self.attention.output_buffer);

            lib.rmsnorm(
                self.feed_forward.input_buffer,
                self.hidden_state,
                weights.rms_ffn_input[(layer * config.dim)..][0..config.dim],
            );

            try self.feed_forward.forward(weights, layer);

            lib.add(self.hidden_state, self.feed_forward.output_buffer);
        }

        lib.rmsnorm(self.hidden_state, self.hidden_state, weights.rms_final);
        lib.matmul(self.logits, self.hidden_state, weights.classifier);
    }
};
