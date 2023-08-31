const std = @import("std");

pub const VectorArray = struct {
    vector_dim: usize,
    data: []const f32,

    pub fn init(vector_dim: usize, data: []const f32) VectorArray {
        std.debug.assert(data.len % vector_dim == 0);

        return VectorArray{ .vector_dim = vector_dim, .data = data };
    }

    pub fn getVector(self: *const VectorArray, index: usize) []const f32 {
        const dim = self.vector_dim;

        return self.data[(index * dim)..][0..dim];
    }
};
