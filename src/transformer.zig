const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Attention = @import("attention.zig");
const Checkpoint = @import("checkpoint.zig");
const Cli = @import("cli.zig");
const FeedForward = @import("feed_forward.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
sequence_length: usize,
attention: Attention,
feed_forward: FeedForward,
hidden_state: []f32,
logits: []f32,

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const checkpoint = try Checkpoint.init(allocator, cli);

    errdefer checkpoint.deinit();

    const sequence_length = if (cli.n_steps == 0) checkpoint.max_sequence_length else cli.n_steps;
    const attention = try Attention.init(allocator, checkpoint, sequence_length);

    errdefer attention.deinit();

    const feed_forward = try FeedForward.init(allocator, checkpoint);

    errdefer feed_forward.deinit();

    const hidden_state = try allocator.alloc(f32, checkpoint.embedding_size);

    errdefer allocator.free(hidden_state);

    const logits = try allocator.alloc(f32, checkpoint.vocab_size);

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .sequence_length = sequence_length,
        .attention = attention,
        .feed_forward = feed_forward,
        .hidden_state = hidden_state,
        .logits = logits,
    };
}

pub fn deinit(self: *const Self) void {
    self.checkpoint.deinit();
    self.attention.deinit();
    self.feed_forward.deinit();
    self.allocator.free(self.hidden_state);
    self.allocator.free(self.logits);
}

pub fn forward(self: *const Self, token: usize, pos: usize) !void {
    const checkpoint = self.checkpoint;
    const weights = checkpoint.weights;

    @memcpy(self.hidden_state, weights.embedding_vectors.at(token));

    for (0..checkpoint.n_layers) |layer| {
        lib.rmsnorm(
            self.hidden_state,
            weights.attention_norm_vectors.at(layer),
            self.attention.input_buffer,
        );

        try self.attention.forward(pos, layer);

        lib.add(self.hidden_state, self.attention.output_buffer);

        lib.rmsnorm(
            self.hidden_state,
            weights.feed_forward_norm_vectors.at(layer),
            self.feed_forward.input_buffer,
        );

        try self.feed_forward.forward(layer);

        lib.add(self.hidden_state, self.feed_forward.output_buffer);
    }

    lib.rmsnorm(
        self.hidden_state,
        weights.final_norm_vector,
        self.hidden_state,
    );

    try weights.classifier_matrices.multiplyVector(0, self.hidden_state, self.logits);
}
