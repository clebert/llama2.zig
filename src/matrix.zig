const Self = @This();

const std = @import("std");
const Vector = @import("vector.zig");

rows: []const Vector,

pub fn readLeaky(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    m_rows: usize,
    n_cols: usize,
) !Self {
    return .{ .rows = try Vector.readMultipleLeaky(allocator, file, m_rows, n_cols) };
}

pub fn readMultipleLeaky(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    n_matrices: usize,
    m_rows: usize,
    n_cols: usize,
) ![]Self {
    const matrices = try allocator.alloc(Self, n_matrices);

    for (matrices) |*matrix| {
        matrix.* = try readLeaky(allocator, file, m_rows, n_cols);
    }

    return matrices;
}

pub fn multiplyVector(self: Self, input: Vector, output: Vector) !void {
    std.debug.assert(self.rows.len == output.values.len);

    for (output.values, 0..) |*value, index| {
        value.* = try self.rows[index].computeScalarProduct(input);
    }
}
