const Self = @This();

const std = @import("std");
const simd = @import("simd.zig");

data: []align(std.atomic.cache_line) f32,

pub fn initLeaky(allocator: std.mem.Allocator, data_size: usize) !Self {
    return .{ .data = try allocator.alignedAlloc(f32, std.atomic.cache_line, data_size) };
}

pub fn initAllLeaky(allocator: std.mem.Allocator, n_vectors: usize, data_size: usize) ![]Self {
    const vectors = try allocator.alloc(Self, n_vectors);

    for (vectors) |*vector| {
        vector.* = try initLeaky(allocator, data_size);
    }

    return vectors;
}

pub fn read(self: Self, file: std.fs.File) !void {
    const data: [*]u8 = @ptrCast(self.data);

    try file.reader().readNoEof(data[0 .. self.data.len * @sizeOf(f32)]);
}

pub fn readAll(file: std.fs.File, vectors: []const Self) !void {
    for (vectors) |vector| {
        try vector.read(file);
    }
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
