const std = @import("std");

const checkpoint = @import("checkpoint.zig");

pub const RunState = struct {
    x: []f32, // dim
    xb: []f32, // dim
    xb2: []f32, // dim
    hb: []f32, // hidden_dim
    hb2: []f32, // hidden_dim
    q: []f32, // dim
    k: []f32, // dim
    v: []f32, // dim
    att: []f32, // n_heads * seq_len
    logits: []f32, // vocab_size
    key_cache: []f32, // n_layers * seq_len * dim
    value_cache: []f32, // n_layers * seq_len * dim
};

pub fn allocRunState(
    allocator: std.mem.Allocator,
    config: checkpoint.Config,
    run_state: *RunState,
) !void {
    run_state.* = RunState{
        .x = try allocator.alloc(f32, config.dim),
        .xb = try allocator.alloc(f32, config.dim),
        .xb2 = try allocator.alloc(f32, config.dim),
        .hb = try allocator.alloc(f32, config.hidden_dim),
        .hb2 = try allocator.alloc(f32, config.hidden_dim),
        .q = try allocator.alloc(f32, config.dim),
        .k = try allocator.alloc(f32, config.dim),
        .v = try allocator.alloc(f32, config.dim),
        .att = try allocator.alloc(f32, config.n_heads * config.seq_len),
        .logits = try allocator.alloc(f32, config.vocab_size),
        .key_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * config.dim),
        .value_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * config.dim),
    };

    // TODO: remove?
    @memset(run_state.x, 0);
    @memset(run_state.xb, 0);
    @memset(run_state.xb2, 0);
    @memset(run_state.hb, 0);
    @memset(run_state.hb2, 0);
    @memset(run_state.q, 0);
    @memset(run_state.k, 0);
    @memset(run_state.v, 0);
    @memset(run_state.att, 0);
    @memset(run_state.logits, 0);
    @memset(run_state.key_cache, 0);
    @memset(run_state.value_cache, 0);
}
