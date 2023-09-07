const std = @import("std");

pub fn VectorArray(comptime T: type) type {
    return struct {
        const Self = @This();

        vector_size: usize,
        data: T,

        pub fn init(vector_size: usize, data: T) Self {
            std.debug.assert(data.len % vector_size == 0);

            return Self{ .vector_size = vector_size, .data = data };
        }

        pub fn at(self: *const Self, vector_index: usize) T {
            return self.data[(vector_index * self.vector_size)..][0..self.vector_size];
        }
    };
}
