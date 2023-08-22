const std = @import("std");

const checkpoint = @import("checkpoint.zig");
const lib = @import("lib.zig");

pub const FeedForward = struct {
    const Self = @This();

    input_buffer: []f32,
    hidden_buffer: []f32,
    residual_buffer: []f32,
    output_buffer: []f32,

    pub fn init(self: *Self, allocator: std.mem.Allocator, config: *const checkpoint.Config) !void {
        self.input_buffer = try allocator.alloc(f32, config.dim);
        self.hidden_buffer = try allocator.alloc(f32, config.hidden_dim);
        self.residual_buffer = try allocator.alloc(f32, config.hidden_dim);
        self.output_buffer = try allocator.alloc(f32, config.dim);
    }

    pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
        allocator.free(self.input_buffer);
        allocator.free(self.hidden_buffer);
        allocator.free(self.residual_buffer);
        allocator.free(self.output_buffer);
    }

    pub fn forward(
        self: *const Self,
        weights: *const checkpoint.Weights,
        layer: usize,
    ) !void {
        const dim = self.input_buffer.len;
        const hidden_dim = self.hidden_buffer.len;

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

        for (0..hidden_dim) |i| {
            self.hidden_buffer[i] = silu(self.hidden_buffer[i]) * self.residual_buffer[i];
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
