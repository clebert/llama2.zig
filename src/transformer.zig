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

fn accum(a: []f32, b: []const f32) void {
    std.debug.assert(a.len == b.len);

    for (a, 0..) |*item, index| {
        item.* += b[index];
    }
}

test "accumulate two slices" {
    var a = [_]f32{ 42, 85 };
    const b = [_]f32{ 100, 200 };
    const expected = [_]f32{ 142, 285 };

    accum(a[0..], b[0..]);

    try std.testing.expectEqualSlices(f32, expected[0..], a[0..]);
}

fn rmsnorm(o: []f32, x: []const f32, weight: []const f32) void {
    std.debug.assert(o.len == x.len);
    std.debug.assert(weight.len >= o.len);

    // calculate sum of squares
    var ss: f32 = 0.0;
    for (x) |item| {
        ss += item * item;
    }
    ss /= @floatFromInt(x.len);
    ss += 1e-5;
    ss = 1.0 / std.math.sqrt(ss);

    // normalize and scale
    for (o, 0..) |*item, i| {
        item.* = weight[i] * (ss * x[i]);
    }
}

test "rms normalization" {
    var o = [_]f32{ 0, 0 };
    const x = [_]f32{ 2, 3 };
    const weight = [_]f32{ 0.5, 0.5 };

    rmsnorm(o[0..], x[0..], weight[0..]);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5) * ((1.0 / std.math.sqrt(13.0 / 2.0 + 1e-5)) * 2), o[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5) * ((1.0 / std.math.sqrt(13.0 / 2.0 + 1e-5)) * 3), o[1], 1e-5);
}

fn softmax(x: []f32) void {
    var max_val = std.mem.max(f32, x);

    // exp and sum
    var sum: f32 = 0.0;

    for (x) |*item| {
        item.* = std.math.exp(item.* - max_val);

        sum += item.*;
    }

    // normalize
    for (x) |*item| {
        item.* /= sum;
    }
}

test "compute softmax" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0 };

    const expected = [_]f32{
        0.02364052406, 0.06426167518, 0.1746813036, 0.474833058,
        0.02364052406, 0.06426167518, 0.1746813036,
    };

    softmax(x[0..]); // TODO: sum should be 1

    for (x, 0..) |item, i| {
        try std.testing.expectApproxEqAbs(expected[i], item, 0.00001);
    }
}

fn matmul(xout: []f32, x: []const f32, w: []const f32) void {
    @setFloatMode(.Optimized);

    std.debug.assert(w.len >= xout.len * x.len);

    for (0..xout.len) |i| {
        xout[i] = 0;

        const i_n = i * x.len;
        var j: usize = 0;

        // https://github.com/karpathy/llama2.c/pull/95
        while (j < x.len) : (j += 4) {
            const a = @Vector(4, f32){
                w[i_n + j],
                w[i_n + j + 1],
                w[i_n + j + 2],
                w[i_n + j + 3],
            };

            const b = @Vector(4, f32){
                x[j],
                x[j + 1],
                x[j + 2],
                x[j + 3],
            };

            xout[i] += @reduce(.Add, a * b);
        }
    }
}

test "matrix multiplication" {
    var xout = [_]f32{ 0, 0 };

    const x = [_]f32{ 3, 4 };
    const w = [_]f32{ 2, 3, 7, 5 };

    // 3 4 * 2 7 = 18 41
    //       3 5

    const expected = [_]f32{ 18, 41 };

    matmul(xout[0..], x[0..], w[0..]);

    try std.testing.expectEqualSlices(f32, expected[0..], xout[0..]);
}

pub fn run(
    token: usize,
    pos: usize,
    p: checkpoint.Config,
    s: *RunState,
    w: *const checkpoint.Weights,
) void {
    // copy the token embedding into x
    @memcpy(s.x, w.token_embedding_table[(token * p.dim)..][0..s.x.len]);

    const head_size = p.dim / p.n_heads;

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    const freq_cis_real_row = w.freq_cis_real[(pos * head_size / 2)..];
    const freq_cis_imag_row = w.freq_cis_imag[(pos * head_size / 2)..];

    // forward all the layers
    for (0..p.n_layers) |layer| {
        // attention rmsnorm
        rmsnorm(s.xb, s.x, w.rms_att_weight[(layer * p.dim)..]);

        // qkv matmuls for this position
        matmul(s.q, s.xb, w.wq[(layer * p.dim * p.dim)..]);
        matmul(s.k, s.xb, w.wk[(layer * p.dim * p.dim)..]);
        matmul(s.v, s.xb, w.wv[(layer * p.dim * p.dim)..]);

        // apply RoPE rotation to the q and k vectors for each head
        for (0..p.n_heads) |head| {
            // get the q and k vectors for this head
            const q = s.q[(head * head_size)..];
            const k = s.k[(head * head_size)..];

            // rotate q and k by the freq_cis_real and freq_cis_imag
            var i: usize = 0;

            while (i < head_size) : (i += 2) {
                const q0 = q[i];
                const q1 = q[i + 1];
                const k0 = k[i];
                const k1 = k[i + 1];
                const fcr = freq_cis_real_row[i / 2];
                const fci = freq_cis_imag_row[i / 2];

                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        const loff = layer * p.seq_len * p.dim; // kv cache layer offset for convenience
        const key_cache_row = s.key_cache[(loff + pos * p.dim)..];
        const value_cache_row = s.value_cache[(loff + pos * p.dim)..];

        @memcpy(key_cache_row[0..s.k.len], s.k);
        @memcpy(value_cache_row[0..s.v.len], s.v);

        // multihead attention. iterate over all heads
        for (0..p.n_heads) |head| {
            // get the query vector for this head
            const q = s.q[(head * head_size)..];
            // attention scores for this head
            const att = s.att[(head * p.seq_len)..];

            // iterate over all timesteps, including the current one
            for (0..(pos + 1)) |t| {
                // get the key vector for this head and at this timestep
                const k = s.key_cache[(loff + t * p.dim + head * head_size)..];

                // calculate the attention score as the dot product of q and k
                var score: f32 = 0;

                for (0..head_size) |i| {
                    score += q[i] * k[i];
                }

                score /= std.math.sqrt(@as(f32, @floatFromInt(head_size)));

                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att[0..(pos + 1)]);

            // weighted sum of the values, store back into xb
            const xb = s.xb[(head * head_size)..][0..head_size];

            @memset(xb, 0);

            for (0..(pos + 1)) |t| {
                // get the value vector for this head and at this timestep
                const v = s.value_cache[(loff + t * p.dim + head * head_size)..];

                // get the attention weight for this timestep
                const a = att[t];

                // accumulate the weighted value into xb
                for (0..head_size) |i| {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s.xb2, s.xb, w.wo[(layer * p.dim * p.dim)..]);

        // residual connection back into x
        accum(s.x, s.xb2);

        // ffn rmsnorm
        rmsnorm(s.xb, s.x, w.rms_ffn_weight[(layer * p.dim)..]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s.hb, s.xb, w.w1[(layer * p.dim * p.hidden_dim)..]);
        matmul(s.hb2, s.xb, w.w3[(layer * p.dim * p.hidden_dim)..]);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (0..p.hidden_dim) |i| {
            s.hb[i] *= 1.0 / (1.0 + std.math.exp(-s.hb[i]));
        }

        // elementwise multiply with w3(x)
        for (0..p.hidden_dim) |i| {
            s.hb[i] *= s.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s.xb, s.hb, w.w2[(layer * p.dim * p.hidden_dim)..]);

        // residual connection
        accum(s.x, s.xb);
    }

    // final rmsnorm
    rmsnorm(s.x, s.x, w.rms_final_weight);

    // classifier into logits
    matmul(s.logits, s.x, w.wcls);
}
