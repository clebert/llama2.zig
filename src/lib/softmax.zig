const std = @import("std");

pub fn softmax(vector: []f32) void {
    @setFloatMode(.Optimized);

    var max_element: f32 = std.mem.max(f32, vector);
    var sum: f32 = 0;

    for (vector) |*element| {
        element.* = std.math.exp(element.* - max_element);
        sum += element.*;
    }

    for (vector) |*element| {
        element.* /= sum;
    }
}
