const Self = @This();

const std = @import("std");
const simd = @import("simd.zig");

values: []f32,

pub fn createLeaky(allocator: std.mem.Allocator, n_values: usize) !Self {
    return .{ .values = try allocator.alignedAlloc(f32, std.atomic.cache_line, n_values) };
}

pub fn createMultipleLeaky(
    allocator: std.mem.Allocator,
    n_vectors: usize,
    n_values: usize,
) ![]Self {
    const vectors = try allocator.alloc(Self, n_vectors);

    for (vectors) |*vector| {
        vector.* = try createLeaky(allocator, n_values);
    }

    return vectors;
}

pub fn readLeaky(allocator: std.mem.Allocator, file: std.fs.File, n_values: usize) !Self {
    const vector = try createLeaky(allocator, n_values);
    const bytes: [*]u8 = @ptrCast(vector.values);

    try file.reader().readNoEof(bytes[0 .. vector.values.len * @sizeOf(f32)]);

    return vector;
}

pub fn readMultipleLeaky(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    n_vectors: usize,
    n_values: usize,
) ![]Self {
    const vectors = try allocator.alloc(Self, n_vectors);

    for (vectors) |*vector| {
        vector.* = try readLeaky(allocator, file, n_values);
    }

    return vectors;
}

pub fn addVector(self: Self, other: Self) !void {
    try simd.computeVectorAddition(self.values, other.values, self.values);
}

pub fn computeRMSNorm(self: Self, weight: Self, output: Self) !void {
    try simd.computeRMSNorm(self.values, weight.values, output.values);
}

pub fn computeScalarProduct(self: Self, other: Self) !f32 {
    return simd.computeScalarProduct(self.values, other.values);
}
