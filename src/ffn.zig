const Self = @This();

const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const Vector = @import("vector.zig");
const Worker = @import("worker.zig");

checkpoint: Checkpoint,
input: Vector,
gate: Vector,
hidden: Vector,
output: Vector,

pub fn initLeaky(allocator: std.mem.Allocator, checkpoint: Checkpoint) !Self {
    return .{
        .checkpoint = checkpoint,
        .input = try Vector.initLeaky(allocator, checkpoint.embedding_size),
        .gate = try Vector.initLeaky(allocator, checkpoint.ffn_hidden_size),
        .hidden = try Vector.initLeaky(allocator, checkpoint.ffn_hidden_size),
        .output = try Vector.initLeaky(allocator, checkpoint.embedding_size),
    };
}

// SwiGLU activation function: https://arxiv.org/abs/2002.05202
pub fn forward(self: Self, layer: usize, workers: []Worker) !void {
    @setFloatMode(.Optimized);

    const gate_weight = self.checkpoint.ffn_gate_weights[layer];
    const up_weight = self.checkpoint.ffn_up_weights[layer];
    const down_weight = self.checkpoint.ffn_down_weights[layer];

    try gate_weight.multiplyVector(self.input, self.gate, workers);
    try up_weight.multiplyVector(self.input, self.hidden, workers);

    for (0..self.checkpoint.ffn_hidden_size) |index| {
        self.hidden.data[index] *= swish(self.gate.data[index]);
    }

    try down_weight.multiplyVector(self.hidden, self.output, workers);
}

// Swish activation function: https://arxiv.org/abs/1710.05941
inline fn swish(x: f32) f32 {
    return x * sigmoid(x);
}

inline fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
