const Self = @This();

const std = @import("std");
const Attention = @import("attention.zig");
const Checkpoint = @import("checkpoint.zig");
const FFN = @import("ffn.zig");
const Vector = @import("vector.zig");
const Worker = @import("worker.zig");

sequence_length: usize,
checkpoint: Checkpoint,
attention: Attention,
ffn: FFN,
hidden: Vector,
output: Vector,

pub fn initLeaky(allocator: std.mem.Allocator, args: anytype) !Self {
    const checkpoint = try Checkpoint.initLeaky(allocator, args);

    const sequence_length = if (args.sequence_length == 0)
        checkpoint.max_sequence_length
    else
        @min(args.sequence_length, checkpoint.max_sequence_length);

    return .{
        .sequence_length = sequence_length,
        .checkpoint = checkpoint,
        .attention = try Attention.initLeaky(allocator, checkpoint, sequence_length),
        .ffn = try FFN.initLeaky(allocator, checkpoint),
        .hidden = try Vector.initLeaky(allocator, checkpoint.embedding_size),
        .output = try Vector.initLeaky(allocator, checkpoint.vocab_size),
    };
}

pub fn forward(self: Self, token: usize, position: usize, workers: []Worker) !void {
    const embedding_weight = self.checkpoint.embedding_weights[token];

    @memcpy(self.hidden.data, embedding_weight.data);

    for (0..self.checkpoint.n_layers) |layer| {
        const attention_norm_weight = self.checkpoint.attention_norm_weights[layer];
        const ffn_norm_weight = self.checkpoint.ffn_norm_weights[layer];

        try self.hidden.computeRMSNorm(attention_norm_weight, self.attention.input);
        try self.attention.forward(layer, position, workers);
        try self.hidden.addVector(self.attention.output);
        try self.hidden.computeRMSNorm(ffn_norm_weight, self.ffn.input);
        try self.ffn.forward(layer, workers);
        try self.hidden.addVector(self.ffn.output);
    }

    try self.hidden.computeRMSNorm(self.checkpoint.output_norm_weight, self.hidden);
    try self.checkpoint.output_weight.multiplyVector(self.hidden, self.output, workers);
}
