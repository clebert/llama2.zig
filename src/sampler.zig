const Self = @This();

const builtin = @import("builtin");
const std = @import("std");
const math = @import("math.zig");

allocator: std.mem.Allocator,
probability_index_pairs_buffer: []ProbabilityIndexPair,
temperature: f32,
top_p: f32,
rng_state: u64,

pub fn init(allocator: std.mem.Allocator, args: anytype, vocab_size: usize) !Self {
    const probability_index_pairs_buffer =
        try allocator.alloc(ProbabilityIndexPair, vocab_size);

    return Self{
        .allocator = allocator,
        .probability_index_pairs_buffer = probability_index_pairs_buffer,
        .temperature = args.temperature,
        .top_p = args.top_p,
        .rng_state = args.random_seed,
    };
}

pub fn deinit(self: Self) void {
    self.allocator.free(self.probability_index_pairs_buffer);
}

pub fn sample(self: *Self, probability_distribution: []f32) usize {
    if (self.temperature == 0) {
        return math.argmax(probability_distribution);
    }

    for (probability_distribution) |*probability| {
        probability.* /= self.temperature;
    }

    math.softmax(probability_distribution);

    if (self.top_p <= 0 or self.top_p >= 1) {
        return self.sampleMultinomial(probability_distribution);
    }

    return self.sampleNucleus(
        probability_distribution,
        self.top_p,
        self.probability_index_pairs_buffer,
    );
}

const tolerance: comptime_float = std.math.sqrt(std.math.floatEps(f32));

fn sampleMultinomial(self: *Self, probability_distribution: []const f32) usize {
    std.debug.assert(probability_distribution.len > 0);

    var cumulative_probability: f32 = 0;

    if (builtin.mode == .Debug) {
        for (probability_distribution) |probability| {
            cumulative_probability += probability;
        }

        std.debug.assert(std.math.approxEqRel(f32, cumulative_probability, 1, tolerance));

        cumulative_probability = 0;
    }

    const probability_threshold = self.random();

    for (probability_distribution, 0..) |probability, index| {
        cumulative_probability += probability;

        if (probability_threshold < cumulative_probability) {
            return index;
        }
    }

    return probability_distribution.len - 1;
}

const ProbabilityIndexPair = struct { probability: f32, index: usize };

// Nucleus sampling: https://arxiv.org/abs/1904.09751
fn sampleNucleus(
    self: *Self,
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
            probability_index_pairs = probability_index_pairs[0 .. index + 1];

            break;
        }
    }

    probability_threshold = self.random() * cumulative_probability;
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

fn random(self: *Self) f32 {
    @setFloatMode(.Optimized);

    return @as(f32, @floatFromInt(self.xorshift() >> 8)) / 16777216;
}

// https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
fn xorshift(self: *Self) u32 {
    self.rng_state ^= self.rng_state >> 12;
    self.rng_state ^= self.rng_state << 25;
    self.rng_state ^= self.rng_state >> 27;

    return @intCast((self.rng_state *% 0x2545F4914F6CDD1D) >> 32);
}
