const Self = @This();

const std = @import("std");
const Checkpoint = @import("checkpoint.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
input_buffer: []f32,
hidden_buffer: []f32,
scaling_buffer: []f32,
output_buffer: []f32,

pub fn init(allocator: std.mem.Allocator, checkpoint: Checkpoint) !Self {
    const embedding_size = checkpoint.embedding_size;
    const input_buffer = try allocator.alloc(f32, embedding_size);

    errdefer allocator.free(input_buffer);

    const intermediate_size = checkpoint.intermediate_size;
    const hidden_buffer = try allocator.alloc(f32, intermediate_size);

    errdefer allocator.free(hidden_buffer);

    const scaling_buffer = try allocator.alloc(f32, intermediate_size);

    errdefer allocator.free(scaling_buffer);

    const output_buffer = try allocator.alloc(f32, embedding_size);

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .input_buffer = input_buffer,
        .hidden_buffer = hidden_buffer,
        .scaling_buffer = scaling_buffer,
        .output_buffer = output_buffer,
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.input_buffer);
    self.allocator.free(self.hidden_buffer);
    self.allocator.free(self.scaling_buffer);
    self.allocator.free(self.output_buffer);
}

pub fn forward(self: *const Self, layer: usize) !void {
    @setFloatMode(.Optimized);

    const checkpoint = self.checkpoint;
    const weights = checkpoint.weights;

    try weights.feed_forward_hidden_matrices.multiplyVector(
        layer,
        self.input_buffer,
        self.hidden_buffer,
    );

    try weights.feed_forward_scaling_matrices.multiplyVector(
        layer,
        self.input_buffer,
        self.scaling_buffer,
    );

    for (0..checkpoint.intermediate_size) |index| {
        self.hidden_buffer[index] = silu(self.hidden_buffer[index]) * self.scaling_buffer[index];
    }

    try weights.feed_forward_output_matrices.multiplyVector(
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
