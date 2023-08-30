const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Attention = @import("attention.zig");
const Checkpoint = @import("checkpoint.zig");
const Cli = @import("cli.zig");
const FeedForward = @import("feed_forward.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
attention: Attention,
feed_forward: FeedForward,
hidden_state_vector: []f32,
logits_vector: []f32,

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const checkpoint = try Checkpoint.init(if (cli.mmap) null else allocator, cli.checkpoint_path);

    errdefer checkpoint.deinit();

    const attention = try Attention.init(allocator, checkpoint, cli.n_steps);

    errdefer attention.deinit();

    const feed_forward = try FeedForward.init(allocator, checkpoint);

    errdefer feed_forward.deinit();

    const hidden_state_vector = try allocator.alloc(f32, checkpoint.dim);

    errdefer allocator.free(hidden_state_vector);

    const logits_vector = try allocator.alloc(f32, checkpoint.vocab_size);

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .attention = attention,
        .feed_forward = feed_forward,
        .hidden_state_vector = hidden_state_vector,
        .logits_vector = logits_vector,
    };
}

pub fn deinit(self: *const Self) void {
    self.checkpoint.deinit();
    self.attention.deinit();
    self.feed_forward.deinit();
    self.allocator.free(self.hidden_state_vector);
    self.allocator.free(self.logits_vector);
}

pub fn forward(self: *const Self, token: usize, pos: usize) !void {
    const checkpoint = self.checkpoint;
    const weights = checkpoint.weights;

    @memcpy(
        self.hidden_state_vector,
        weights.token_embedding_vectors.getVector(token),
    );

    for (0..checkpoint.n_layers) |layer| {
        lib.rmsnorm(
            self.attention.input_buffer,
            self.hidden_state_vector,
            weights.attention_norm_vectors.getVector(layer),
        );

        try self.attention.forward(pos, layer);

        lib.add(self.hidden_state_vector, self.attention.output_buffer);

        lib.rmsnorm(
            self.feed_forward.input_buffer,
            self.hidden_state_vector,
            weights.feed_forward_norm_vectors.getVector(layer),
        );

        try self.feed_forward.forward(layer);

        lib.add(self.hidden_state_vector, self.feed_forward.output_buffer);
    }

    lib.rmsnorm(self.hidden_state_vector, self.hidden_state_vector, weights.final_norm_vector);

    weights.classifier_matrix.multiplyVector(self.hidden_state_vector, self.logits_vector);
}
