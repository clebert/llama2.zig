const Self = @This();

const std = @import("std");

vector_dim: usize,
data: []const f32,

pub fn init(vector_dim: usize, data: []const f32) Self {
    std.debug.assert(data.len % vector_dim == 0);

    return Self{ .vector_dim = vector_dim, .data = data };
}

pub fn at(self: *const Self, index: usize) []const f32 {
    const dim = self.vector_dim;

    return self.data[(index * dim)..][0..dim];
}
