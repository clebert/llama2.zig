const Self = @This();

const std = @import("std");

dim: usize,
data: []const f32,

pub fn init(dim: usize, data: []const f32) Self {
    std.debug.assert(data.len % dim == 0);

    return Self{ .dim = dim, .data = data };
}

pub fn at(self: *const Self, index: usize) []const f32 {
    return self.data[(index * self.dim)..][0..self.dim];
}
