const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Attention = @import("attention.zig");
const Checkpoint = @import("checkpoint.zig");
const Cli = @import("cli.zig");
const Ffn = @import("ffn.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
sequence_length: usize,
attention: Attention,
ffn: Ffn,
hidden_state: []f32,
logits: []f32,

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const checkpoint = try Checkpoint.init(allocator, cli);

    errdefer checkpoint.deinit();

    const sequence_length = if (cli.n_steps == 0) checkpoint.max_sequence_length else cli.n_steps;
    const attention = try Attention.init(allocator, &checkpoint, sequence_length);

    errdefer attention.deinit();

    const ffn = try Ffn.init(allocator, &checkpoint);

    errdefer ffn.deinit();

    const hidden_state = try allocator.alloc(f32, checkpoint.embedding_size);

    errdefer allocator.free(hidden_state);

    const logits = try allocator.alloc(f32, checkpoint.vocab_size);

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .sequence_length = sequence_length,
        .attention = attention,
        .ffn = ffn,
        .hidden_state = hidden_state,
        .logits = logits,
    };
}

pub fn deinit(self: *const Self) void {
    self.checkpoint.deinit();
    self.attention.deinit();
    self.ffn.deinit();
    self.allocator.free(self.hidden_state);
    self.allocator.free(self.logits);
}

pub fn forward(self: *const Self, token: usize, position: usize) !void {
    const n_layers = self.checkpoint.n_layers;
    const weights = self.checkpoint.weights;

    @memcpy(self.hidden_state, weights.token_embedding_vectors.at(token));

    for (0..n_layers) |layer| {
        const attention_norm_vector = weights.attention_norm_vectors.at(layer);
        const ffn_norm_vector = weights.ffn_norm_vectors.at(layer);

        lib.rmsnorm(self.hidden_state, attention_norm_vector, self.attention.input_vector);

        try self.attention.forward(layer, position);

        lib.add(self.hidden_state, self.attention.output_vector);

        lib.rmsnorm(self.hidden_state, ffn_norm_vector, self.ffn.input_buffer);

        try self.ffn.forward(layer);

        lib.add(self.hidden_state, self.ffn.output_buffer);
    }

    lib.rmsnorm(self.hidden_state, weights.final_norm_vector, self.hidden_state);

    weights.final_classifier_projection_matrix.multiplyVector(self.hidden_state, self.logits);
}
