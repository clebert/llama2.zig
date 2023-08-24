const std = @import("std");

const Checkpoint = @import("checkpoint.zig").Checkpoint;
const lib = @import("lib.zig");

pub const FeedForward = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    checkpoint: *const Checkpoint,

    input_buffer: []f32,
    hidden_buffer: []f32,
    residual_buffer: []f32,
    output_buffer: []f32,

    pub fn init(self: *Self, allocator: std.mem.Allocator, checkpoint: *const Checkpoint) !void {
        self.allocator = allocator;
        self.checkpoint = checkpoint;

        const dim = checkpoint.dim;
        const hidden_dim = checkpoint.hidden_dim;

        self.input_buffer = try allocator.alloc(f32, dim);
        self.hidden_buffer = try allocator.alloc(f32, hidden_dim);
        self.residual_buffer = try allocator.alloc(f32, hidden_dim);
        self.output_buffer = try allocator.alloc(f32, dim);
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
        const dim = checkpoint.dim;
        const hidden_dim = checkpoint.hidden_dim;
        const weights = checkpoint.weights;

        const weights_size = dim * hidden_dim;
        const weights_offset = layer * weights_size;

        const input_to_hidden = weights.ffn_input_to_hidden[weights_offset..][0..weights_size];
        const input_to_residual = weights.ffn_input_to_residual[weights_offset..][0..weights_size];
        const hidden_to_output = weights.ffn_hidden_to_output[weights_offset..][0..weights_size];

        try lib.matmul2(
            .{ self.hidden_buffer, self.input_buffer, input_to_hidden },
            .{ self.residual_buffer, self.input_buffer, input_to_residual },
            dim >= 4096,
        );

        for (0..hidden_dim) |index| {
            self.hidden_buffer[index] =
                silu(self.hidden_buffer[index]) * self.residual_buffer[index];
        }

        lib.matmul(self.output_buffer, self.hidden_buffer, hidden_to_output);
    }
};

// GLU Variants Improve Transformer (https://arxiv.org/abs/2002.05202)
inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

inline fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
