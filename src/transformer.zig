const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Attention = @import("attention.zig");
const Checkpoint = @import("checkpoint.zig");
const Cli = @import("cli.zig");
const FeedForward = @import("feed_forward.zig");

allocator: std.mem.Allocator,
checkpoint: Checkpoint,
attention: Attention,
feed_forward: FeedForward,
hidden_state: []f32,
logits: []f32,

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const checkpoint = try Checkpoint.init(if (cli.mmap) null else allocator, cli.checkpoint_path);

    return Self{
        .allocator = allocator,
        .checkpoint = checkpoint,
        .attention = try Attention.init(allocator, checkpoint, cli.n_steps),
        .feed_forward = try FeedForward.init(allocator, checkpoint),
        .hidden_state = try allocator.alloc(f32, checkpoint.dim),
        .logits = try allocator.alloc(f32, checkpoint.vocab_size),
    };
}

pub fn deinit(self: *const Self) void {
    self.checkpoint.deinit();
    self.attention.deinit();
    self.feed_forward.deinit();
    self.allocator.free(self.hidden_state);
    self.allocator.free(self.logits);
}

pub fn forward(self: *const Self, token: usize, pos: usize) !void {
    const checkpoint = self.checkpoint;
    const dim = checkpoint.dim;
    const weights = checkpoint.weights;

    @memcpy(
        self.hidden_state,
        weights.token_embedding[(token * dim)..][0..self.hidden_state.len],
    );

    for (0..checkpoint.n_layers) |layer| {
        lib.rmsnorm(
            self.attention.input_buffer,
            self.hidden_state,
            weights.attention_input_rms[(layer * dim)..][0..dim],
        );

        try self.attention.forward(pos, layer);

        lib.add(self.hidden_state, self.attention.output_buffer);

        lib.rmsnorm(
            self.feed_forward.input_buffer,
            self.hidden_state,
            weights.ffn_input_rms[(layer * dim)..][0..dim],
        );

        try self.feed_forward.forward(layer);

        lib.add(self.hidden_state, self.feed_forward.output_buffer);
    }

    lib.rmsnorm(self.hidden_state, self.hidden_state, weights.final_rms);
    lib.matmul(self.logits, self.hidden_state, weights.classifier);
}
