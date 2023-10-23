const std = @import("std");

pub fn argmax(data: []f32) usize {
    var max_index: usize = 0;
    var max_element: f32 = data[max_index];

    for (1..data.len) |index| {
        const element = data[index];

        if (element > max_element) {
            max_index = index;
            max_element = element;
        }
    }

    return max_index;
}

pub fn softmax(data: []f32) void {
    @setFloatMode(.Optimized);

    var max_element: f32 = std.mem.max(f32, data);
    var sum: f32 = 0;

    for (data) |*element| {
        element.* = std.math.exp(element.* - max_element);
        sum += element.*;
    }

    for (data) |*element| {
        element.* /= sum;
    }
}
