const std = @import("std");

const checkpoint = @import("checkpoint.zig");
const utils = @import("utils.zig");

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
        @setFloatMode(.Optimized);

        const input_buffer = self.input_buffer;
        const hidden_buffer = self.hidden_buffer;
        const residual_buffer = self.residual_buffer;
        const output_buffer = self.output_buffer;

        std.debug.assert(input_buffer.len == output_buffer.len);
        std.debug.assert(hidden_buffer.len == residual_buffer.len);

        const dim = input_buffer.len;
        const hidden_dim = hidden_buffer.len;
        const weights_size = dim * hidden_dim;
        const weights_offset = layer * weights_size;
        const input_to_hidden = weights.ffn_input_to_hidden[weights_offset..][0..weights_size];
        const input_to_residual = weights.ffn_input_to_residual[weights_offset..][0..weights_size];
        const hidden_to_output = weights.ffn_hidden_to_output[weights_offset..][0..weights_size];

        try matmul2(
            .{ hidden_buffer, input_buffer, input_to_hidden },
            .{ residual_buffer, input_buffer, input_to_residual },
            dim >= 4096,
        );

        for (0..hidden_dim) |i| {
            hidden_buffer[i] = silu(hidden_buffer[i]) * residual_buffer[i];
        }

        utils.matmul(output_buffer, hidden_buffer, hidden_to_output);
    }
};

// https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

inline fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

fn matmul2(args_1: anytype, args_2: anytype, multi_threaded: bool) !void {
    const cpu_count = std.Thread.getCpuCount() catch 1;

    if (multi_threaded and cpu_count > 2) {
        const thread_1 = try std.Thread.spawn(.{}, utils.matmul, args_1);
        const thread_2 = try std.Thread.spawn(.{}, utils.matmul, args_2);

        thread_1.join();
        thread_2.join();
    } else {
        @call(.auto, utils.matmul, args_1);
        @call(.auto, utils.matmul, args_2);
    }
}
