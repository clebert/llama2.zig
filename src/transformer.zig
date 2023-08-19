const std = @import("std");

const checkpoint = @import("checkpoint.zig");
const utils = @import("utils.zig");

pub const RunState = struct {
    hidden_state: []f32,

    attention_ffn_input_buffer: []f32,
    attention_ffn_output_buffer: []f32,

    attention_scores: []f32,
    query_buffer: []f32,
    key_buffer: []f32,
    value_buffer: []f32,
    key_cache: []f32,
    value_cache: []f32,

    ffn_weighted_input_buffer_1: []f32,
    ffn_weighted_input_buffer_2: []f32,

    logits: []f32,
};

pub fn allocRunState(
    allocator: std.mem.Allocator,
    config: checkpoint.Config,
    run_state: *RunState,
) !void {
    const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

    run_state.* = RunState{
        .hidden_state = try allocator.alloc(f32, config.dim),

        .attention_ffn_input_buffer = try allocator.alloc(f32, config.dim),
        .attention_ffn_output_buffer = try allocator.alloc(f32, config.dim),

        .attention_scores = try allocator.alloc(f32, config.n_heads * config.seq_len),
        .query_buffer = try allocator.alloc(f32, config.dim),
        .key_buffer = try allocator.alloc(f32, kv_dim),
        .value_buffer = try allocator.alloc(f32, kv_dim),
        .key_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim),
        .value_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim),

        .ffn_weighted_input_buffer_1 = try allocator.alloc(f32, config.hidden_dim),
        .ffn_weighted_input_buffer_2 = try allocator.alloc(f32, config.hidden_dim),

        .logits = try allocator.alloc(f32, config.vocab_size),
    };
}

pub fn decode(
    allocator: std.mem.Allocator,
    token: usize,
    pos: usize,
    config: checkpoint.Config,
    run_state: *RunState,
    weights: *const checkpoint.Weights,
) !void {
    @setFloatMode(.Optimized);

    // copy the token embedding into hidden_state
    @memcpy(
        run_state.hidden_state,
        weights.token_embedding_table[(token * config.dim)..][0..run_state.hidden_state.len],
    );

    const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    // integer multiplier of the kv sharing in multiquery
    const kv_mul = config.n_heads / config.n_kv_heads;
    const head_size = config.dim / config.n_heads;
    const head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(head_size)));

    // forward all the layers
    for (0..config.n_layers) |layer| {
        // attention rmsnorm
        utils.rmsnorm(run_state.attention_ffn_input_buffer, run_state.hidden_state, weights.rms_att_weight[(layer * config.dim)..]);

        const dim_multithreading_threshold = 4096;

        var pool: std.Thread.Pool = undefined;

        // qkv matmuls for this position
        if (config.dim >= dim_multithreading_threshold) {
            try pool.init(std.Thread.Pool.Options{
                .allocator = allocator,
                .n_jobs = @max(1, @min(3, std.Thread.getCpuCount() catch 1)),
            });

            defer pool.deinit();

            try pool.spawn(utils.matmul, .{
                run_state.query_buffer,
                run_state.attention_ffn_input_buffer,
                weights.wq[(layer * config.dim * config.dim)..],
            });

            try pool.spawn(utils.matmul, .{
                run_state.key_buffer,
                run_state.attention_ffn_input_buffer,
                weights.wk[(layer * config.dim * kv_dim)..],
            });

            try pool.spawn(utils.matmul, .{
                run_state.value_buffer,
                run_state.attention_ffn_input_buffer,
                weights.wv[(layer * config.dim * kv_dim)..],
            });
        } else {
            utils.matmul(
                run_state.query_buffer,
                run_state.attention_ffn_input_buffer,
                weights.wq[(layer * config.dim * config.dim)..],
            );

            utils.matmul(
                run_state.key_buffer,
                run_state.attention_ffn_input_buffer,
                weights.wk[(layer * config.dim * kv_dim)..],
            );

            utils.matmul(
                run_state.value_buffer,
                run_state.attention_ffn_input_buffer,
                weights.wv[(layer * config.dim * kv_dim)..],
            );
        }

        rope(pos, head_size, kv_dim, &config, run_state);

        // save key,value at this time step (pos) to our kv cache
        const loff = layer * config.seq_len * kv_dim; // kv cache layer offset for convenience
        const key_cache_row = run_state.key_cache[(loff + pos * kv_dim)..];
        const value_cache_row = run_state.value_cache[(loff + pos * kv_dim)..];

        @memcpy(key_cache_row[0..run_state.key_buffer.len], run_state.key_buffer);
        @memcpy(value_cache_row[0..run_state.value_buffer.len], run_state.value_buffer);

        // multihead attention. iterate over all heads
        for (0..config.n_heads) |head| {
            // get the query vector for this head
            const q = run_state.query_buffer[(head * head_size)..];
            // attention scores for this head
            const att = run_state.attention_scores[(head * config.seq_len)..];

            // iterate over all timesteps, including the current one
            for (0..(pos + 1)) |t| {
                // get the key vector for this head and at this timestep
                const k = run_state.key_cache[(loff + t * kv_dim + (head / kv_mul) * head_size)..];

                // calculate the attention score as the dot product of q and k
                var score: f32 = 0;

                for (0..head_size) |i| {
                    score += q[i] * k[i];
                }

                score /= head_size_sqrt;

                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            utils.softmax(att[0..(pos + 1)]);

            // weighted sum of the values, store back into intermediate_buffer
            const intermediate_buffer =
                run_state.attention_ffn_input_buffer[(head * head_size)..][0..head_size];

            @memset(intermediate_buffer, 0);

            for (0..(pos + 1)) |t| {
                // get the value vector for this head and at this timestep
                const v = run_state.value_cache[(loff + t * kv_dim + (head / kv_mul) * head_size)..];

                // get the attention weight for this timestep
                const a = att[t];

                // accumulate the weighted value into intermediate_buffer
                for (0..head_size) |i| {
                    intermediate_buffer[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        utils.matmul(
            run_state.attention_ffn_output_buffer,
            run_state.attention_ffn_input_buffer,
            weights.wo[(layer * config.dim * config.dim)..],
        );

        // residual connection back into hidden_state
        utils.accum(run_state.hidden_state, run_state.attention_ffn_output_buffer);

        // ffn rmsnorm
        utils.rmsnorm(
            run_state.attention_ffn_input_buffer,
            run_state.hidden_state,
            weights.rms_ffn_weight[(layer * config.dim)..],
        );

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        if (config.dim >= dim_multithreading_threshold) {
            try pool.init(std.Thread.Pool.Options{
                .allocator = allocator,
                .n_jobs = @max(1, @min(2, std.Thread.getCpuCount() catch 1)),
            });

            defer pool.deinit();

            try pool.spawn(utils.matmul, .{
                run_state.ffn_weighted_input_buffer_1,
                run_state.attention_ffn_input_buffer,
                weights.w1[(layer * config.dim * config.hidden_dim)..],
            });

            try pool.spawn(utils.matmul, .{
                run_state.ffn_weighted_input_buffer_2,
                run_state.attention_ffn_input_buffer,
                weights.w3[(layer * config.dim * config.hidden_dim)..],
            });
        } else {
            utils.matmul(
                run_state.ffn_weighted_input_buffer_1,
                run_state.attention_ffn_input_buffer,
                weights.w1[(layer * config.dim * config.hidden_dim)..][0..(config.dim * config.hidden_dim)],
            );

            utils.matmul(
                run_state.ffn_weighted_input_buffer_2,
                run_state.attention_ffn_input_buffer,
                weights.w3[(layer * config.dim * config.hidden_dim)..][0..(config.dim * config.hidden_dim)],
            );
        }

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (run_state.ffn_weighted_input_buffer_1) |*s| {
            s.* *= 1 / (1 + std.math.exp(-s.*));
        }

        // elementwise multiply with w3(x)
        for (0..config.hidden_dim) |i| {
            run_state.ffn_weighted_input_buffer_1[i] *= run_state.ffn_weighted_input_buffer_2[i];
        }

        // final matmul to get the output of the ffn
        utils.matmul(
            run_state.attention_ffn_output_buffer,
            run_state.ffn_weighted_input_buffer_1,
            weights.w2[(layer * config.dim * config.hidden_dim)..][0..(config.dim * config.hidden_dim)],
        );

        // residual connection
        utils.accum(run_state.hidden_state, run_state.attention_ffn_output_buffer);
    }

    // final rmsnorm
    utils.rmsnorm(run_state.hidden_state, run_state.hidden_state, weights.rms_final_weight);

    // classifier into logits
    utils.matmul(run_state.logits, run_state.hidden_state, weights.wcls);
}

pub fn rope(
    pos: usize,
    head_size: usize,
    kv_dim: usize,
    config: *const checkpoint.Config,
    run_state: *RunState,
) void {
    @setFloatMode(.Optimized);

    var i: usize = 0;

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    // https://github.com/karpathy/llama2.c/issues/302#issue-1851956882
    // https://github.com/karpathy/llama2.c/commit/bd182289c596fa6059eb7b3b7c8ccd04b5c90fc3
    while (i < config.dim) : (i += 2) {
        const head_dim: f32 = @floatFromInt(i % head_size);
        const freq: f32 = 1 / std.math.pow(f32, 10000, head_dim / @as(f32, @floatFromInt(head_size)));
        const value: f32 = @as(f32, @floatFromInt(pos)) * freq;
        const fcr: f32 = std.math.cos(value);
        const fci: f32 = std.math.sin(value);

        // rotate q
        const q0 = run_state.query_buffer[i];
        const q1 = run_state.query_buffer[i + 1];

        run_state.query_buffer[i] = q0 * fcr - q1 * fci;
        run_state.query_buffer[i + 1] = q0 * fci + q1 * fcr;

        // rotate k
        if (i < kv_dim) {
            const k0 = run_state.key_buffer[i];
            const k1 = run_state.key_buffer[i + 1];

            run_state.key_buffer[i] = k0 * fcr - k1 * fci;
            run_state.key_buffer[i + 1] = k0 * fci + k1 * fcr;
        }
    }
}
