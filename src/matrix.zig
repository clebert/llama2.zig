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

const max_thread_count = 24;

pub fn multiplyVector(self: Self, input: Vector, output: Vector) !void {
    if (self.thread_count == 0) {
        try computeMatrixVectorMultiplication(self.rows, input, output.values);

        return;
    }

    const n_threads = @min(max_thread_count, self.thread_count);
    const thread_chunk_size = output.values.len / n_threads;

    var threads: [max_thread_count]std.Thread = undefined;

    for (threads[0..n_threads], 0..) |*thread, index| {
        thread.* = try std.Thread.spawn(.{}, computeMatrixVectorMultiplication, .{
            self.rows[index * thread_chunk_size ..][0..thread_chunk_size],
            input,
            output.values[index * thread_chunk_size ..][0..thread_chunk_size],
        });
    }

    for (threads[0..n_threads]) |thread| {
        thread.join();
    }

    if (output.values.len % n_threads > 0) {
        try computeMatrixVectorMultiplication(
            self.rows[n_threads * thread_chunk_size ..],
            input,
            output.values[n_threads * thread_chunk_size ..],
        );
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
