const build_options = @import("build_options");
const std = @import("std");
const dot = @import("dot.zig").dot;

extern fn accelerateMulMatrixVector(
    row_major_matrix: [*c]const f32,
    input_vector: [*c]const f32,
    output_vector: [*c]f32,
    m_rows: i64,
    n_cols: i64,
) void;

extern fn metalMulMatrixVector(
    row_major_matrix: [*c]const f32,
    input_vector: [*c]const f32,
    output_vector: [*c]f32,
    m_rows: u64,
    n_cols: u64,
) void;

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

pub const MatrixArray = struct {
    matrix_m_rows: usize,
    matrix_n_cols: usize,
    row_major_data: []const f32,

    pub fn init(
        matrix_m_rows: usize,
        matrix_n_cols: usize,
        row_major_data: []const f32,
    ) MatrixArray {
        const matrix_dim = matrix_m_rows * matrix_n_cols;

        std.debug.assert(row_major_data.len % matrix_dim == 0);

        return MatrixArray{
            .matrix_m_rows = matrix_m_rows,
            .matrix_n_cols = matrix_n_cols,
            .row_major_data = row_major_data,
        };
    }

    pub fn getMatrix(self: *const MatrixArray, index: usize) Matrix {
        const m_rows = self.matrix_m_rows;
        const n_cols = self.matrix_n_cols;
        const dim = m_rows * n_cols;

        return Matrix.init(m_rows, n_cols, self.row_major_data[(index * dim)..][0..dim]);
    }
};

pub const Matrix = struct {
    m_rows: usize,
    n_cols: usize,

    // 0x0 0x1 ... 0xN
    // 1x0 1x1 ... 1xN
    // ... ... ... ...
    // Mx0 Mx1 ... MxN

    // 0x0, 0x1, 0xN, 1x0 1x1, 1xN, Mx0, Mx1, MxN
    row_major_data: []const f32,

    pub fn init(m_rows: usize, n_cols: usize, row_major_data: []const f32) Matrix {
        std.debug.assert(m_rows * n_cols == row_major_data.len);

        return Matrix{ .m_rows = m_rows, .n_cols = n_cols, .row_major_data = row_major_data };
    }

    pub fn multiplyVector(
        self: *const Matrix,
        input_vector: []const f32,
        output_vector: []f32,
    ) void {
        std.debug.assert(input_vector.len == self.n_cols);
        std.debug.assert(output_vector.len == self.m_rows);

        if (build_options.accelerate) {
            accelerateMulMatrixVector(
                self.row_major_data.ptr,
                input_vector.ptr,
                output_vector.ptr,
                @intCast(self.m_rows),
                @intCast(self.n_cols),
            );
        } else if (build_options.metal) {
            metalMulMatrixVector(
                self.row_major_data.ptr,
                input_vector.ptr,
                output_vector.ptr,
                self.m_rows,
                self.n_cols,
            );
        } else {
            for (output_vector, 0..) |*element, index| {
                element.* = dot(
                    self.row_major_data[(index * self.n_cols)..][0..self.n_cols],
                    input_vector,
                );
            }
        }
    }

    pub fn multiplyVector2(args_1: anytype, args_2: anytype, multi_threaded: bool) !void {
        const cpu_count = std.Thread.getCpuCount() catch 1;

        if (!build_options.metal and multi_threaded and cpu_count > 2) {
            const thread_1 = try std.Thread.spawn(.{}, Matrix.multiplyVector, args_1);
            const thread_2 = try std.Thread.spawn(.{}, Matrix.multiplyVector, args_2);

            thread_1.join();
            thread_2.join();
        } else {
            @call(.auto, Matrix.multiplyVector, args_1);
            @call(.auto, Matrix.multiplyVector, args_2);
        }
    }

    pub fn multiplyVector3(
        args_1: anytype,
        args_2: anytype,
        args_3: anytype,
        multi_threaded: bool,
    ) !void {
        const cpu_count = std.Thread.getCpuCount() catch 1;

        if (!build_options.metal and multi_threaded and cpu_count > 2) {
            const thread_1 = try std.Thread.spawn(.{}, Matrix.multiplyVector, args_1);
            const thread_2 = try std.Thread.spawn(.{}, Matrix.multiplyVector, args_2);
            const thread_3 = try std.Thread.spawn(.{}, Matrix.multiplyVector, args_3);

            thread_1.join();
            thread_2.join();
            thread_3.join();
        } else {
            @call(.auto, Matrix.multiplyVector, args_1);
            @call(.auto, Matrix.multiplyVector, args_2);
            @call(.auto, Matrix.multiplyVector, args_3);
        }
    }
};
