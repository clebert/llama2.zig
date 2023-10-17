const Self = @This();

const std = @import("std");
const Attention = @import("attention.zig");
const Checkpoint = @import("checkpoint.zig");
const FFN = @import("ffn.zig");
const Tensor = @import("./tensor.zig").Tensor;
const vector = @import("vector.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
sequence_length: usize,
attention: Attention,
ffn: FFN,
hidden_buffer: Tensor(1),
output_buffer: Tensor(1),

pub fn init(allocator: std.mem.Allocator, model_path: []const u8, n_steps: usize) !Self {
    const checkpoint = try Checkpoint.init(allocator, model_path);

    errdefer checkpoint.deinit();

    const sequence_length = if (n_steps == 0) checkpoint.max_sequence_length else n_steps;
    const attention = try Attention.init(allocator, checkpoint, sequence_length);

    errdefer attention.deinit();

    const ffn = try FFN.init(allocator, checkpoint);

    errdefer ffn.deinit();

    const hidden_buffer = try Tensor(1).init(allocator, [_]usize{checkpoint.embedding_size});

    errdefer hidden_buffer.deinit();

    const output_buffer = try Tensor(1).init(allocator, [_]usize{checkpoint.vocab_size});

    errdefer output_buffer.deinit();

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .sequence_length = sequence_length,
        .attention = attention,
        .ffn = ffn,
        .hidden_buffer = hidden_buffer,
        .output_buffer = output_buffer,
    };
}

pub fn deinit(self: *const Self) void {
    self.checkpoint.deinit();
    self.attention.deinit();
    self.ffn.deinit();
    self.hidden_buffer.deinit();
    self.output_buffer.deinit();
}

pub fn forward(self: *const Self, token: usize, position: usize) !void {
    const weights = self.checkpoint.weights;

    @memcpy(self.hidden_buffer.data, weights.token_embedding_vectors.slice(token).data);

    for (0..self.checkpoint.n_layers) |layer| {
        const attention_norm_vector = weights.attention_norm_vectors.slice(layer);
        const ffn_norm_vector = weights.ffn_norm_vectors.slice(layer);

        vector.rmsnorm(
            self.hidden_buffer.data,
            attention_norm_vector.data,
            self.attention.input_buffer.data,
        );

        try self.attention.forward(layer, position);

        vector.add(self.hidden_buffer.data, self.attention.output_buffer.data);
        vector.rmsnorm(self.hidden_buffer.data, ffn_norm_vector.data, self.ffn.input_buffer.data);

        try self.ffn.forward(layer);

        vector.add(self.hidden_buffer.data, self.ffn.output_buffer.data);
    }

    vector.rmsnorm(
        self.hidden_buffer.data,
        weights.output_norm_vector.data,
        self.hidden_buffer.data,
    );

    weights.output_matrix.multiplyVector(self.hidden_buffer, self.output_buffer);
}
