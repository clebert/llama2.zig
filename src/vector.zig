const Self = @This();

const std = @import("std");
const simd = @import("simd.zig");

data: []align(std.atomic.cache_line) f32,

pub fn createLeaky(allocator: std.mem.Allocator, data_size: usize) !Self {
    return .{ .data = try allocator.alignedAlloc(f32, std.atomic.cache_line, data_size) };
}

pub fn createMultipleLeaky(
    allocator: std.mem.Allocator,
    n_vectors: usize,
    data_size: usize,
) ![]Self {
    const vectors = try allocator.alloc(Self, n_vectors);

    for (vectors) |*vector| {
        vector.* = try createLeaky(allocator, data_size);
    }

    return vectors;
}

pub fn readLeaky(allocator: std.mem.Allocator, file: std.fs.File, data_size: usize) !Self {
    const vector = try createLeaky(allocator, data_size);
    const data: [*]u8 = @ptrCast(vector.data);

    try file.reader().readNoEof(data[0 .. vector.data.len * @sizeOf(f32)]);

    return vector;
}

pub fn readMultipleLeaky(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    n_vectors: usize,
    data_size: usize,
) ![]Self {
    const vectors = try allocator.alloc(Self, n_vectors);

    for (vectors) |*vector| {
        vector.* = try readLeaky(allocator, file, data_size);
    }

    return vectors;
}

pub fn addVector(self: Self, other: Self) !void {
    try simd.computeVectorAddition(self.data, other.data, self.data);
}

pub fn computeRMSNorm(self: Self, weight: Self, output: Self) !void {
    try simd.computeRMSNorm(self.data, weight.data, output.data);
}

pub fn computeScalarProduct(self: Self, other: Self) !f32 {
    return simd.computeScalarProduct(self.data, other.data);
}
