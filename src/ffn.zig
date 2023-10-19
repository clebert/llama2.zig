const Self = @This();

const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const Tensor = @import("./tensor.zig").Tensor;

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
input_buffer: Tensor(1),
gate_buffer: Tensor(1),
hidden_buffer: Tensor(1),
output_buffer: Tensor(1),

pub fn init(allocator: std.mem.Allocator, checkpoint: Checkpoint) !Self {
    const input_buffer = try Tensor(1).init(allocator, [_]usize{checkpoint.embedding_size});

    errdefer input_buffer.deinit();

    const gate_buffer = try Tensor(1).init(allocator, [_]usize{checkpoint.ffn_hidden_size});

    errdefer gate_buffer.deinit();

    const hidden_buffer = try Tensor(1).init(allocator, [_]usize{checkpoint.ffn_hidden_size});

    errdefer hidden_buffer.deinit();

    const output_buffer = try Tensor(1).init(allocator, [_]usize{checkpoint.embedding_size});

    errdefer output_buffer.deinit();

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .input_buffer = input_buffer,
        .gate_buffer = gate_buffer,
        .hidden_buffer = hidden_buffer,
        .output_buffer = output_buffer,
    };
}

pub fn deinit(self: *const Self) void {
    self.input_buffer.deinit();
    self.gate_buffer.deinit();
    self.hidden_buffer.deinit();
    self.output_buffer.deinit();
}

// SwiGLU activation function: https://arxiv.org/abs/2002.05202
pub fn forward(self: *const Self, layer: usize) void {
    @setFloatMode(.Optimized);

    const weights = self.checkpoint.weights;
    const gate_matrix = weights.ffn_gate_matrices.slice(layer);
    const up_matrix = weights.ffn_up_matrices.slice(layer);
    const down_matrix = weights.ffn_down_matrices.slice(layer);

    gate_matrix.computeMatrixVectorMultiplication(self.input_buffer, self.gate_buffer);
    up_matrix.computeMatrixVectorMultiplication(self.input_buffer, self.hidden_buffer);

    for (0..self.checkpoint.ffn_hidden_size) |index| {
        self.hidden_buffer.values[index] *= swish(self.gate_buffer.values[index]);
    }

    down_matrix.computeMatrixVectorMultiplication(self.hidden_buffer, self.output_buffer);
}

// Swish activation function: https://arxiv.org/abs/1710.05941
inline fn swish(x: f32) f32 {
    return x * sigmoid(x);
}

inline fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
