const Self = @This();

const std = @import("std");
const Vector = @import("vector.zig");

rows: []const Vector,
thread_count: usize,

pub fn readLeaky(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    m_rows: usize,
    n_cols: usize,
    thread_count: usize,
) !Self {
    return .{
        .rows = try Vector.readMultipleLeaky(allocator, file, m_rows, n_cols),
        .thread_count = thread_count,
    };
}

pub fn readMultipleLeaky(
    allocator: std.mem.Allocator,
    file: std.fs.File,
    n_matrices: usize,
    m_rows: usize,
    n_cols: usize,
    thread_count: usize,
) ![]Self {
    const matrices = try allocator.alloc(Self, n_matrices);

    for (matrices) |*matrix| {
        matrix.* = try readLeaky(allocator, file, m_rows, n_cols, thread_count);
    }

    return matrices;
}

const max_thread_count = 8;

pub fn multiplyVector(self: Self, input: Vector, output: Vector) !void {
    if (self.thread_count == 0) {
        try computeMatrixVectorMultiplication(self.rows, input, output.values);

        return;
    }

    const n_threads = @min(try std.Thread.getCpuCount(), max_thread_count, self.thread_count);

    if (output.values.len % n_threads != 0) {
        return error.UnsupportedThreadCount;
    }

    const partial_length = output.values.len / n_threads;

    var threads: [max_thread_count]std.Thread = undefined;

    for (threads[0..n_threads], 0..) |*thread, index| {
        thread.* = try std.Thread.spawn(.{}, computeMatrixVectorMultiplication, .{
            self.rows[index * partial_length .. (index + 1) * partial_length],
            input,
            output.values[index * partial_length .. (index + 1) * partial_length],
        });
    }

    for (threads[0..n_threads]) |thread| {
        thread.join();
    }
}

fn computeMatrixVectorMultiplication(
    rows: []const Vector,
    input: Vector,
    output_values: []f32,
) !void {
    std.debug.assert(rows.len == output_values.len);

    for (output_values, 0..) |*value, index| {
        value.* = try rows[index].computeScalarProduct(input);
    }
}
