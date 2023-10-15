const Self = @This();

const std = @import("std");
const Attention = @import("attention.zig");
const Checkpoint = @import("checkpoint.zig");
const Cli = @import("cli.zig");
const FeedForward = @import("feed_forward.zig");
const Tensor = @import("./tensor.zig").Tensor;
const vector = @import("vector.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
sequence_length: usize,
attention: Attention,
feed_forward: FeedForward,
hidden_buffer: Tensor(1),
logits_buffer: Tensor(1),

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const checkpoint = try Checkpoint.init(allocator, cli);

    errdefer checkpoint.deinit();

    const sequence_length = if (cli.n_steps == 0) checkpoint.max_sequence_length else cli.n_steps;
    const attention = try Attention.init(allocator, checkpoint, sequence_length);

    errdefer attention.deinit();

    const feed_forward = try FeedForward.init(allocator, checkpoint);

    errdefer feed_forward.deinit();

    const hidden_buffer = try Tensor(1).init(allocator, [_]usize{checkpoint.embedding_size});

    errdefer hidden_buffer.deinit();

    const logits_buffer = try Tensor(1).init(allocator, [_]usize{checkpoint.vocab_size});

    errdefer logits_buffer.deinit();

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .sequence_length = sequence_length,
        .attention = attention,
        .feed_forward = feed_forward,
        .hidden_buffer = hidden_buffer,
        .logits_buffer = logits_buffer,
    };
}

pub fn deinit(self: *const Self) void {
    self.checkpoint.deinit();
    self.attention.deinit();
    self.feed_forward.deinit();
    self.hidden_buffer.deinit();
    self.logits_buffer.deinit();
}

pub fn forward(self: *const Self, token: usize, position: usize) !void {
    const weights = self.checkpoint.weights;

    @memcpy(self.hidden_buffer.data, weights.token_embedding_vectors.slice(token).data);

    for (0..self.checkpoint.n_layers) |layer| {
        const attention_pre_norm_vector = weights.attention_pre_norm_vectors.slice(layer);
        const feed_forward_pre_norm_vector = weights.feed_forward_pre_norm_vectors.slice(layer);

        vector.rmsnorm(
            self.hidden_buffer.data,
            attention_pre_norm_vector.data,
            self.attention.input_buffer.data,
        );

        try self.attention.forward(layer, position);

        vector.add(self.hidden_buffer.data, self.attention.output_buffer.data);

        vector.rmsnorm(
            self.hidden_buffer.data,
            feed_forward_pre_norm_vector.data,
            self.feed_forward.input_buffer.data,
        );

        try self.feed_forward.forward(layer);

        vector.add(self.hidden_buffer.data, self.feed_forward.output_buffer.data);
    }

    vector.rmsnorm(
        self.hidden_buffer.data,
        weights.classifier_pre_norm_vector.data,
        self.hidden_buffer.data,
    );

    weights.classifier_matrix.multiplyVector(self.hidden_buffer, self.logits_buffer);
}
