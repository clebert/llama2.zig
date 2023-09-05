const Self = @This();

const std = @import("std");

size: usize,
data: []const f32,

pub fn init(size: usize, data: []const f32) Self {
    std.debug.assert(data.len % size == 0);

    return Self{ .size = size, .data = data };
}

pub fn at(self: *const Self, index: usize) []const f32 {
    return self.data[(index * self.size)..][0..self.size];
}
