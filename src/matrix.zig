const Self = @This();

const std = @import("std");
const vector = @import("vector.zig");

m_rows: usize,
n_cols: usize,

// 0x0 0x1 ... 0xN
// 1x0 1x1 ... 1xN
// ... ... ... ...
// Mx0 Mx1 ... MxN
//
// => 0x0, 0x1, 0xN, 1x0 1x1, 1xN, Mx0, Mx1, MxN
row_major_data: []const f32,

pub fn init(m_rows: usize, n_cols: usize, row_major_data: []const f32) Self {
    const matrix_size = m_rows * n_cols;

    std.debug.assert(row_major_data.len == matrix_size);

    return Self{ .m_rows = m_rows, .n_cols = n_cols, .row_major_data = row_major_data };
}

pub fn slice(
    allocator: std.mem.Allocator,
    m_rows: usize,
    n_cols: usize,
    row_major_data: []const f32,
) ![]Self {
    const matrix_size = m_rows * n_cols;

    std.debug.assert(row_major_data.len % matrix_size == 0);

    var matrices = try allocator.alloc(Self, row_major_data.len / matrix_size);

    for (matrices, 0..) |*matrix, matrix_index| {
        matrix.* = Self.init(
            m_rows,
            n_cols,
            row_major_data[(matrix_index * matrix_size)..][0..matrix_size],
        );
    }

    return matrices;
}

pub fn multiplyVector(self: *const Self, input_vector: []const f32, output_vector: []f32) void {
    const m_rows = self.m_rows;
    const n_cols = self.n_cols;
    const row_major_data = self.row_major_data;

    std.debug.assert(input_vector.len == n_cols);
    std.debug.assert(output_vector.len == m_rows);

    for (output_vector, 0..) |*element, row| {
        element.* = vector.dot(row_major_data[(row * n_cols)..][0..n_cols], input_vector);
    }
}
