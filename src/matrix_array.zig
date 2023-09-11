const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Matrix = @import("./matrix.zig");

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

    std.debug.assert(row_major_data.len % matrix_size == 0);

    return Self{ .m_rows = m_rows, .n_cols = n_cols, .row_major_data = row_major_data };
}

pub fn at(self: *const Self, matrix_index: usize) Matrix {
    const m_rows = self.m_rows;
    const n_cols = self.n_cols;
    const matrix_size = m_rows * n_cols;

    return Matrix.init(
        m_rows,
        n_cols,
        self.row_major_data[(matrix_index * matrix_size)..][0..matrix_size],
    );
}
