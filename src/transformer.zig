const std = @import("std");

const Attention = @import("attention.zig").Attention;
const Checkpoint = @import("checkpoint.zig").Checkpoint;
const FeedForward = @import("feed_forward.zig").FeedForward;
const lib = @import("lib.zig");

pub const Transformer = struct {
    const Self = @This();

    checkpoint: *const Checkpoint,

    hidden_state: []f32,
    logits: []f32,
    attention: Attention,
    feed_forward: FeedForward,

    pub fn init(self: *Self, allocator: std.mem.Allocator, checkpoint: *const Checkpoint) !void {
        self.checkpoint = checkpoint;
        self.hidden_state = try allocator.alloc(f32, checkpoint.dim);
        self.logits = try allocator.alloc(f32, checkpoint.vocab_size);

        try self.attention.init(allocator, checkpoint);
        try self.feed_forward.init(allocator, checkpoint);
    }

    pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
        allocator.free(self.hidden_state);
        allocator.free(self.logits);

        self.attention.deinit(allocator);
        self.feed_forward.deinit(allocator);
    }

    pub fn forward(self: *const Self, token: usize, pos: usize) !void {
        const checkpoint = self.checkpoint;
        const dim = checkpoint.dim;
        const weights = checkpoint.weights;

        @memcpy(
            self.hidden_state,
            weights.token_embedding[(token * dim)..][0..self.hidden_state.len],
        );

        for (0..checkpoint.n_layers) |layer| {
            lib.rmsnorm(
                self.attention.input_buffer,
                self.hidden_state,
                weights.attention_input_rms[(layer * dim)..][0..dim],
            );

            try self.attention.forward(pos, layer);

            lib.add(self.hidden_state, self.attention.output_buffer);

            lib.rmsnorm(
                self.feed_forward.input_buffer,
                self.hidden_state,
                weights.ffn_input_rms[(layer * dim)..][0..dim],
            );

            try self.feed_forward.forward(layer);

            lib.add(self.hidden_state, self.feed_forward.output_buffer);
        }

        lib.rmsnorm(self.hidden_state, self.hidden_state, weights.final_rms);
        lib.matmul(self.logits, self.hidden_state, weights.classifier);
    }
};
