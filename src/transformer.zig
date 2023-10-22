const Self = @This();

const std = @import("std");
const Attention = @import("attention.zig");
const Checkpoint = @import("checkpoint.zig");
const FFN = @import("ffn.zig");
const Vector = @import("vector.zig");

checkpoint: Checkpoint,
sequence_length: usize,
attention: Attention,
ffn: FFN,
hidden: Vector,
output: Vector,

pub fn createLeaky(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    custom_sequence_length: usize,
) !Self {
    const checkpoint = try Checkpoint.readLeaky(allocator, model_path);

    const sequence_length = if (custom_sequence_length == 0)
        checkpoint.max_sequence_length
    else
        @min(custom_sequence_length, checkpoint.max_sequence_length);

    return .{
        .checkpoint = checkpoint,
        .sequence_length = sequence_length,
        .attention = try Attention.createLeaky(allocator, checkpoint, sequence_length),
        .ffn = try FFN.createLeaky(allocator, checkpoint),
        .hidden = try Vector.createLeaky(allocator, checkpoint.embedding_size),
        .output = try Vector.createLeaky(allocator, checkpoint.vocab_size),
    };
}

pub fn forward(self: Self, token: usize, position: usize) !void {
    const embedding_weight = self.checkpoint.embedding_weights[token];

    @memcpy(self.hidden.values, embedding_weight.values);

    for (0..self.checkpoint.n_layers) |layer| {
        const attention_norm_weight = self.checkpoint.attention_norm_weights[layer];
        const ffn_norm_weight = self.checkpoint.ffn_norm_weights[layer];

        try self.hidden.computeRMSNorm(attention_norm_weight, self.attention.input);
        try self.attention.forward(layer, position);
        try self.hidden.addVector(self.attention.output);
        try self.hidden.computeRMSNorm(ffn_norm_weight, self.ffn.input);
        try self.ffn.forward(layer);
        try self.hidden.addVector(self.ffn.output);
    }

    try self.hidden.computeRMSNorm(self.checkpoint.output_norm_weight, self.hidden);
    try self.checkpoint.output_weight.multiplyVector(self.hidden, self.output);
}
