const std = @import("std");

pub fn sample(rng_value: f32, probabilities: []f32) usize {
    std.debug.assert(probabilities.len > 0);

    var cumulative_probability: f32 = 0;

    for (probabilities, 0..) |probability, index| {
        cumulative_probability += probability;

        if (rng_value < cumulative_probability) {
            return index;
        }
    }

    return probabilities.len - 1;
}
