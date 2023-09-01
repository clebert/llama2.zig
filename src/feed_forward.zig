const Self = @This();

const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const matrix = @import("matrix.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
input_buffer: []f32,
hidden_buffer: []f32,
residual_buffer: []f32,
output_buffer: []f32,

pub fn init(allocator: std.mem.Allocator, checkpoint: Checkpoint) !Self {
    const dim = checkpoint.dim;
    const hidden_dim = checkpoint.hidden_dim;

    const input_buffer = try allocator.alloc(f32, dim);

    errdefer allocator.free(input_buffer);

    const hidden_buffer = try allocator.alloc(f32, hidden_dim);

    errdefer allocator.free(hidden_buffer);

    const residual_buffer = try allocator.alloc(f32, hidden_dim);

    errdefer allocator.free(residual_buffer);

    const output_buffer = try allocator.alloc(f32, dim);

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .input_buffer = input_buffer,
        .hidden_buffer = hidden_buffer,
        .residual_buffer = residual_buffer,
        .output_buffer = output_buffer,
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.input_buffer);
    self.allocator.free(self.hidden_buffer);
    self.allocator.free(self.residual_buffer);
    self.allocator.free(self.output_buffer);
}

pub fn forward(self: *const Self, layer: usize) !void {
    @setFloatMode(.Optimized);

    const checkpoint = self.checkpoint;
    const hidden_dim = checkpoint.hidden_dim;
    const weights = checkpoint.weights;

    try weights.feed_forward_hidden_matrix.multiplyVector(
        layer,
        self.input_buffer,
        self.hidden_buffer,
    );

    try weights.feed_forward_residual_matrix.multiplyVector(
        layer,
        self.input_buffer,
        self.residual_buffer,
    );

    for (0..hidden_dim) |index| {
        self.hidden_buffer[index] = silu(self.hidden_buffer[index]) * self.residual_buffer[index];
    }

    try weights.feed_forward_output_matrix.multiplyVector(
        layer,
        self.hidden_buffer,
        self.output_buffer,
    );
}

// GLU Variants Improve Transformer (https://arxiv.org/abs/2002.05202)
inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

inline fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
