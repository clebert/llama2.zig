const std = @import("std");

pub fn argmax(v: []f32) usize {
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
