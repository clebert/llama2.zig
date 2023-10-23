const Self = @This();

const std = @import("std");
const Vector = @import("vector.zig");
const Worker = @import("worker.zig");

rows: []const Vector,

pub fn initLeaky(allocator: std.mem.Allocator, m_rows: usize, n_cols: usize) !Self {
    return .{ .rows = try Vector.initAllLeaky(allocator, m_rows, n_cols) };
}

pub fn initAllLeaky(
    allocator: std.mem.Allocator,
    n_matrices: usize,
    m_rows: usize,
    n_cols: usize,
) ![]Self {
    const matrices = try allocator.alloc(Self, n_matrices);

    for (matrices) |*matrix| {
        matrix.* = try initLeaky(allocator, m_rows, n_cols);
    }

    return matrices;
}

pub fn read(self: Self, file: std.fs.File) !void {
    for (self.rows) |row| {
        try row.read(file);
    }
}

pub fn readAll(file: std.fs.File, matrices: []const Self) !void {
    for (matrices) |matrix| {
        try matrix.read(file);
    }
}

pub fn multiplyVector(self: Self, input: Vector, output: Vector, workers: []Worker) !void {
    if (workers.len == 0) {
        try Worker.MatrixVectorMultiplication.run(.{
            .rows = self.rows,
            .input = input,
            .output_data = output.data,
        });
    } else {
        const chunk_size = output.data.len / workers.len;

        for (workers, 0..) |*worker, index| {
            worker.schedule(.{
                .rows = self.rows[index * chunk_size ..][0..chunk_size],
                .input = input,
                .output_data = output.data[index * chunk_size ..][0..chunk_size],
            });
        }

        if (output.data.len % workers.len > 0) {
            try Worker.MatrixVectorMultiplication.run(.{
                .rows = self.rows[workers.len * chunk_size ..],
                .input = input,
                .output_data = output.data[workers.len * chunk_size ..],
            });
        }

        for (workers) |*worker| {
            worker.wait();
        }
    }
}
