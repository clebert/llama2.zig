const std = @import("std");

const Attention = @import("attention.zig").Attention;
const checkpoint = @import("checkpoint.zig");
const FeedForward = @import("feed_forward.zig").FeedForward;
const lib = @import("lib.zig");
const utils = @import("utils.zig");

pub const RunState = struct {
    hidden_state: []f32,
    logits: []f32,
};

pub fn allocRunState(
    allocator: std.mem.Allocator,
    config: checkpoint.Config,
    run_state: *RunState,
) !void {
    run_state.* = RunState{
        .hidden_state = try allocator.alloc(f32, config.dim),
        .logits = try allocator.alloc(f32, config.vocab_size),
    };
}

pub fn decode(
    token: usize,
    pos: usize,
    config: checkpoint.Config,
    run_state: *RunState,
    weights: *const checkpoint.Weights,
    attention: *Attention,
    feed_forward: *FeedForward,
) !void {
    @setFloatMode(.Optimized);

    // copy the token embedding into hidden_state
    @memcpy(
        run_state.hidden_state,
        weights.token_embedding[(token * config.dim)..][0..run_state.hidden_state.len],
    );

    // forward all the layers
    for (0..config.n_layers) |layer| {
        // attention rmsnorm
        utils.rmsnorm(
            attention.input_buffer,
            run_state.hidden_state,
            weights.rms_attention_input[(layer * config.dim)..],
        );

        try attention.forward(&config, weights, pos, layer);

        // residual connection back into hidden_state
        utils.accum(run_state.hidden_state, attention.output_buffer);

        // ffn rmsnorm
        utils.rmsnorm(
            feed_forward.input_buffer,
            run_state.hidden_state,
            weights.rms_ffn_input[(layer * config.dim)..],
        );

        try feed_forward.forward(weights, layer);

        // residual connection
        utils.accum(run_state.hidden_state, feed_forward.output_buffer);
    }

    // final rmsnorm
    utils.rmsnorm(run_state.hidden_state, run_state.hidden_state, weights.rms_final);

    // classifier into logits
    lib.matmul(run_state.logits, run_state.hidden_state, weights.classifier);
}
