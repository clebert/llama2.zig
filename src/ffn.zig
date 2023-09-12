const Self = @This();

const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const Matrix = @import("./matrix.zig");

allocator: std.mem.Allocator,

hidden_size: usize,

hidden_projection_matrices: []const Matrix,
scaling_projection_matrices: []const Matrix,
output_projection_matrices: []const Matrix,

input_buffer: []f32,
hidden_buffer: []f32,
scaling_buffer: []f32,
output_buffer: []f32,

pub fn init(allocator: std.mem.Allocator, checkpoint: *const Checkpoint) !Self {
    const embedding_size = checkpoint.embedding_size;
    const hidden_size = checkpoint.hidden_size;
    const weights = checkpoint.weights;

    const input_buffer = try allocator.alloc(f32, embedding_size);

    errdefer allocator.free(input_buffer);

    const hidden_buffer = try allocator.alloc(f32, hidden_size);

    errdefer allocator.free(hidden_buffer);

    const scaling_buffer = try allocator.alloc(f32, hidden_size);

    errdefer allocator.free(scaling_buffer);

    const output_buffer = try allocator.alloc(f32, embedding_size);

    return Self{
        .allocator = allocator,

        .hidden_size = hidden_size,

        .hidden_projection_matrices = weights.ffn_hidden_projection_matrices,
        .scaling_projection_matrices = weights.ffn_scaling_projection_matrices,
        .output_projection_matrices = weights.ffn_output_projection_matrices,

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

    const hidden_projection_matrix = self.hidden_projection_matrices[layer];
    const scaling_projection_matrix = self.scaling_projection_matrices[layer];
    const output_projection_matrix = self.output_projection_matrices[layer];

    hidden_projection_matrix.multiplyVector(self.input_buffer, self.hidden_buffer);
    scaling_projection_matrix.multiplyVector(self.input_buffer, self.scaling_buffer);

    for (0..self.hidden_size) |index| {
        self.hidden_buffer[index] = silu(self.hidden_buffer[index]) * self.scaling_buffer[index];
    }

    output_projection_matrix.multiplyVector(self.hidden_buffer, self.output_buffer);
}

// GLU Variants Improve Transformer (https://arxiv.org/abs/2002.05202)
inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

inline fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
