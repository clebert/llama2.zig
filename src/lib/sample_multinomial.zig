const builtin = @import("builtin");
const std = @import("std");

const tolerance: comptime_float = std.math.sqrt(std.math.floatEps(f32));

pub fn sampleMultinomial(probability_threshold: f32, probability_distribution: []const f32) usize {
    std.debug.assert(probability_distribution.len > 0);

    var cumulative_probability: f32 = 0;

    if (builtin.mode == .Debug) {
        for (probability_distribution) |probability| {
            cumulative_probability += probability;
        }

        std.debug.assert(std.math.approxEqRel(f32, cumulative_probability, 1, tolerance));

        cumulative_probability = 0;
    }

    for (probability_distribution, 0..) |probability, index| {
        cumulative_probability += probability;

        if (probability_threshold < cumulative_probability) {
            return index;
        }
    }

    return probability_distribution.len - 1;
}
