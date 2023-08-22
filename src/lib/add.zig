const std = @import("std");

pub fn add(a: []f32, b: []const f32) void {
    @setFloatMode(.Optimized);

    std.debug.assert(a.len == b.len);

    for (a, 0..) |*element, index| {
        element.* += b[index];
    }
}
