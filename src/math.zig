const std = @import("std");

pub fn argmax(values: []f32) usize {
    var max_index: usize = 0;
    var max_value: f32 = values[max_index];

    for (1..values.len) |index| {
        const value = values[index];

        if (value > max_value) {
            max_index = index;
            max_value = value;
        }
    }

    return max_index;
}

pub fn softmax(values: []f32) void {
    @setFloatMode(.Optimized);

    var max_value: f32 = std.mem.max(f32, values);
    var sum: f32 = 0;

    for (values) |*value| {
        value.* = std.math.exp(value.* - max_value);
        sum += value.*;
    }

    for (values) |*value| {
        value.* /= sum;
    }
}
