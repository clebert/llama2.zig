const std = @import("std");

pub const ProbabilityIndexPair = struct {
    probability: f32,
    index: usize,
};

// The Curious Case of Neural Text Degeneration (https://arxiv.org/abs/1904.09751)
pub fn sampleNucleus(
    rng_value: f32,
    probability_distribution: []const f32,
    top_p: f32,
    probability_index_pairs_buffer: []ProbabilityIndexPair,
) usize {
    @setFloatMode(.Optimized);

    std.debug.assert(probability_distribution.len > 0);

    // https://github.com/karpathy/llama2.c/commit/d421a95b2bfe593b2d9e5c147f3efc8d128afe0e
    var probability_threshold: f32 =
        (1 - top_p) / @as(f32, @floatFromInt(probability_distribution.len - 1));

    var n_probability_index_pairs: usize = 0;

    for (probability_distribution, 0..) |probability, index| {
        if (probability_threshold < probability) {
            probability_index_pairs_buffer[n_probability_index_pairs].probability = probability;
            probability_index_pairs_buffer[n_probability_index_pairs].index = index;
            n_probability_index_pairs += 1;
        }
    }

    var probability_index_pairs = probability_index_pairs_buffer[0..n_probability_index_pairs];

    std.sort.block(ProbabilityIndexPair, probability_index_pairs, {}, lessThan);

    var cumulative_probability: f32 = 0;

    for (probability_index_pairs, 0..) |probability_index_pair, index| {
        cumulative_probability += probability_index_pair.probability;

        if (cumulative_probability > top_p) {
            probability_index_pairs = probability_index_pairs[0..(index + 1)];

            break;
        }
    }

    probability_threshold = rng_value * cumulative_probability;
    cumulative_probability = 0;

    for (probability_index_pairs) |probability_index_pair| {
        cumulative_probability += probability_index_pair.probability;

        if (probability_threshold < cumulative_probability) {
            return probability_index_pair.index;
        }
    }

    return probability_index_pairs[probability_index_pairs.len - 1].index;
}

fn lessThan(context: void, lhs: ProbabilityIndexPair, rhs: ProbabilityIndexPair) bool {
    _ = context;

    return rhs.probability < lhs.probability;
}
