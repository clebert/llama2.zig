const std = @import("std");

pub fn argmax(v: []f32) usize {
    @setFloatMode(.Optimized);

    // return argmax of v in elements 0..n
    var max_i: usize = 0;
    var max_p: f32 = v[0];

    for (1..v.len) |i| {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }

    return max_i;
}

pub fn sample(rng: *std.rand.DefaultPrng, probabilities: []f32) usize {
    @setFloatMode(.Optimized);

    var r = rng.random().float(f32);
    var cdf: f32 = 0.0;

    for (probabilities, 0..) |probability, i| {
        cdf += probability;

        if (r < cdf) {
            return i;
        }
    }

    return probabilities.len - 1;
}

// struct used when sorting probabilities during top-p sampling
pub const ProbIndex = struct { prob: f32, index: usize };

pub fn sampleTopP(
    rng: *std.rand.DefaultPrng,
    probabilities: []f32,
    top_p: f32,
    prob_indices_buffer: []ProbIndex,
) usize {
    @setFloatMode(.Optimized);

    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".

    // elements smaller than (1 - top_p) / (probabilities.len - 1) cannot be part of the result
    // and can be filtered out directly
    // https://github.com/karpathy/llama2.c/commit/d421a95b2bfe593b2d9e5c147f3efc8d128afe0e
    const cutoff: f32 = (1 - top_p) / @as(f32, @floatFromInt(probabilities.len - 1));

    var n0: usize = 0;

    for (probabilities, 0..) |prob, index| {
        if (prob >= cutoff) {
            prob_indices_buffer[n0].prob = prob;
            prob_indices_buffer[n0].index = index;
            n0 += 1;
        }
    }

    var filtered_prob_indices = prob_indices_buffer[0..n0];

    // sort indices in descending order of probabilities
    std.sort.block(ProbIndex, filtered_prob_indices, {}, lessThan);

    // truncate the list where cumulative probability exceeds topp
    var cumulative_prob: f32 = 0;
    var truncated_prob_indices: ?[]ProbIndex = null;

    for (filtered_prob_indices, 0..) |prob_index, index| {
        cumulative_prob += prob_index.prob;

        if (cumulative_prob > top_p) {
            truncated_prob_indices = filtered_prob_indices[0..(index + 1)];

            break; // we've exceeded topp by including index
        }
    }

    // sample from the truncated list
    var r = rng.random().float(f32) * cumulative_prob;
    var cdf: f32 = 0.0;

    if (truncated_prob_indices) |prob_indices| {
        for (prob_indices) |prob_index| {
            cdf += prob_index.prob;

            if (r < cdf) {
                return prob_index.index;
            }
        }
    }

    return filtered_prob_indices[filtered_prob_indices.len - 1].index;
}

fn lessThan(context: void, lhs: ProbIndex, rhs: ProbIndex) bool {
    _ = context;

    return rhs.prob < lhs.prob;
}

pub fn softmax(x: []f32) void {
    @setFloatMode(.Optimized);

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

pub fn rmsnorm(o: []f32, x: []const f32, weight: []const f32) void {
    @setFloatMode(.Optimized);

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

pub fn accum(a: []f32, b: []const f32) void {
    @setFloatMode(.Optimized);

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
