const std = @import("std");

const checkpoint = @import("checkpoint.zig");
const utils = @import("utils.zig");

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
    const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

    run_state.* = RunState{
        .x = try allocator.alloc(f32, config.dim),
        .xb = try allocator.alloc(f32, config.dim),
        .xb2 = try allocator.alloc(f32, config.dim),
        .hb = try allocator.alloc(f32, config.hidden_dim),
        .hb2 = try allocator.alloc(f32, config.hidden_dim),
        .q = try allocator.alloc(f32, config.dim),
        .k = try allocator.alloc(f32, kv_dim),
        .v = try allocator.alloc(f32, kv_dim),
        .att = try allocator.alloc(f32, config.n_heads * config.seq_len),
        .logits = try allocator.alloc(f32, config.vocab_size),
        .key_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim),
        .value_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * kv_dim),
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

    // copy the token embedding into x
    @memcpy(run_state.x, weights.token_embedding_table[(token * config.dim)..][0..run_state.x.len]);

    const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    // integer multiplier of the kv sharing in multiquery
    const kv_mul = config.n_heads / config.n_kv_heads;
    const head_size = config.dim / config.n_heads;
    const head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(head_size)));

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    const freq_cis_real_row = weights.freq_cis_real[(pos * head_size / 2)..];
    const freq_cis_imag_row = weights.freq_cis_imag[(pos * head_size / 2)..];

    // forward all the layers
    for (0..config.n_layers) |layer| {
        // attention rmsnorm
        utils.rmsnorm(run_state.xb, run_state.x, weights.rms_att_weight[(layer * config.dim)..]);

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
                run_state.q,
                run_state.xb,
                weights.wq[(layer * config.dim * config.dim)..],
            });

            try pool.spawn(utils.matmul, .{
                run_state.k,
                run_state.xb,
                weights.wk[(layer * config.dim * kv_dim)..],
            });

            try pool.spawn(utils.matmul, .{
                run_state.v,
                run_state.xb,
                weights.wv[(layer * config.dim * kv_dim)..],
            });
        } else {
            utils.matmul(run_state.q, run_state.xb, weights.wq[(layer * config.dim * config.dim)..]);
            utils.matmul(run_state.k, run_state.xb, weights.wk[(layer * config.dim * kv_dim)..]);
            utils.matmul(run_state.v, run_state.xb, weights.wv[(layer * config.dim * kv_dim)..]);
        }

        var dim_i: usize = 0;

        // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
        while (dim_i < config.dim) : (dim_i += 2) {
            const q0 = run_state.q[dim_i];
            const q1 = run_state.q[dim_i + 1];
            const fcr = freq_cis_real_row[(dim_i % head_size) / 2];
            const fci = freq_cis_imag_row[(dim_i % head_size) / 2];

            run_state.q[dim_i] = q0 * fcr - q1 * fci;
            run_state.q[dim_i + 1] = q0 * fci + q1 * fcr;
        }

        dim_i = 0;

        while (dim_i < kv_dim) : (dim_i += 2) {
            const k0 = run_state.k[dim_i];
            const k1 = run_state.k[dim_i + 1];
            const fcr = freq_cis_real_row[(dim_i % head_size) / 2];
            const fci = freq_cis_imag_row[(dim_i % head_size) / 2];

            run_state.k[dim_i] = k0 * fcr - k1 * fci;
            run_state.k[dim_i + 1] = k0 * fci + k1 * fcr;
        }

        // save key,value at this time step (pos) to our kv cache
        const loff = layer * config.seq_len * kv_dim; // kv cache layer offset for convenience
        const key_cache_row = run_state.key_cache[(loff + pos * kv_dim)..];
        const value_cache_row = run_state.value_cache[(loff + pos * kv_dim)..];

        @memcpy(key_cache_row[0..run_state.k.len], run_state.k);
        @memcpy(value_cache_row[0..run_state.v.len], run_state.v);

        // multihead attention. iterate over all heads
        for (0..config.n_heads) |head| {
            // get the query vector for this head
            const q = run_state.q[(head * head_size)..];
            // attention scores for this head
            const att = run_state.att[(head * config.seq_len)..];

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

            // weighted sum of the values, store back into xb
            const xb = run_state.xb[(head * head_size)..][0..head_size];

            @memset(xb, 0);

            for (0..(pos + 1)) |t| {
                // get the value vector for this head and at this timestep
                const v = run_state.value_cache[(loff + t * kv_dim + (head / kv_mul) * head_size)..];

                // get the attention weight for this timestep
                const a = att[t];

                // accumulate the weighted value into xb
                for (0..head_size) |i| {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        utils.matmul(run_state.xb2, run_state.xb, weights.wo[(layer * config.dim * config.dim)..]);

        // residual connection back into x
        utils.accum(run_state.x, run_state.xb2);

        // ffn rmsnorm
        utils.rmsnorm(run_state.xb, run_state.x, weights.rms_ffn_weight[(layer * config.dim)..]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        if (config.dim >= dim_multithreading_threshold) {
            try pool.init(std.Thread.Pool.Options{
                .allocator = allocator,
                .n_jobs = @max(1, @min(2, std.Thread.getCpuCount() catch 1)),
            });

            defer pool.deinit();

            try pool.spawn(utils.matmul, .{
                run_state.hb,
                run_state.xb,
                weights.w1[(layer * config.dim * config.hidden_dim)..],
            });

            try pool.spawn(utils.matmul, .{
                run_state.hb2,
                run_state.xb,
                weights.w3[(layer * config.dim * config.hidden_dim)..],
            });
        } else {
            utils.matmul(
                run_state.hb,
                run_state.xb,
                weights.w1[(layer * config.dim * config.hidden_dim)..],
            );

            utils.matmul(
                run_state.hb2,
                run_state.xb,
                weights.w3[(layer * config.dim * config.hidden_dim)..],
            );
        }

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (0..config.hidden_dim) |i| {
            run_state.hb[i] *= 1.0 / (1.0 + std.math.exp(-run_state.hb[i]));
        }

        // elementwise multiply with w3(x)
        for (0..config.hidden_dim) |i| {
            run_state.hb[i] *= run_state.hb2[i];
        }

        // final matmul to get the output of the ffn
        utils.matmul(
            run_state.xb,
            run_state.hb,
            weights.w2[(layer * config.dim * config.hidden_dim)..],
        );

        // residual connection
        utils.accum(run_state.x, run_state.xb);
    }

    // final rmsnorm
    utils.rmsnorm(run_state.x, run_state.x, weights.rms_final_weight);

    // classifier into logits
    utils.matmul(run_state.logits, run_state.x, weights.wcls);
}
