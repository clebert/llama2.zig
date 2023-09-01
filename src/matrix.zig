const Self = @This();

const build_options = @import("build_options");
const std = @import("std");
const lib = @import("lib.zig");

extern fn matvecmulAccelerate(
    row_major_matrix: [*c]const f32,
    input_vector: [*c]const f32,
    output_vector: [*c]f32,
    m_rows: i64,
    n_cols: i64,
) void;

extern fn matvecmulMetal(
    row_major_matrix: [*c]const f32,
    input_vector: [*c]const f32,
    output_vector: [*c]f32,
    m_rows: u64,
    n_cols: u64,
) void;

allocator: std.mem.Allocator,
m_rows: usize,
m_rows_main_thread: usize,
m_rows_worker_thread: usize,
n_cols: usize,

// 0x0 0x1 ... 0xN
// 1x0 1x1 ... 1xN
// ... ... ... ...
// Mx0 Mx1 ... MxN
//
// => 0x0, 0x1, 0xN, 1x0 1x1, 1xN, Mx0, Mx1, MxN
row_major_data: []const f32,

worker_threads: []std.Thread,

pub fn init(
    allocator: std.mem.Allocator,
    multithreading: bool,
    m_rows: usize,
    n_cols: usize,
    row_major_data: []const f32,
) !Self {
    const matrix_dim = m_rows * n_cols;

    std.debug.assert(row_major_data.len % matrix_dim == 0);

    const n_worker_threads = if (!multithreading or build_options.accelerate or build_options.metal)
        0
    else
        std.Thread.getCpuCount() catch 1;

    const m_rows_main_thread = if (n_worker_threads > 0) m_rows % n_worker_threads else m_rows;
    const m_rows_worker_thread = if (n_worker_threads > 0) m_rows / n_worker_threads else 0;

    return Self{
        .allocator = allocator,
        .m_rows = m_rows,
        .m_rows_main_thread = m_rows_main_thread,
        .m_rows_worker_thread = m_rows_worker_thread,
        .n_cols = n_cols,
        .row_major_data = row_major_data,
        .worker_threads = try allocator.alloc(std.Thread, n_worker_threads),
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.worker_threads);
}

pub fn multiplyVector(
    self: *const Self,
    matrix_index: usize,
    input_vector: []const f32,
    output_vector: []f32,
) !void {
    const m_rows = self.m_rows;
    const n_cols = self.n_cols;

    std.debug.assert(input_vector.len == n_cols);
    std.debug.assert(output_vector.len == m_rows);

    const matrix_dim = m_rows * n_cols;
    const row_major_data = self.row_major_data[(matrix_index * matrix_dim)..][0..matrix_dim];

    if (build_options.accelerate) {
        matvecmulAccelerate(
            row_major_data.ptr,
            input_vector.ptr,
            output_vector.ptr,
            @intCast(m_rows),
            @intCast(n_cols),
        );
    } else if (build_options.metal) {
        matvecmulMetal(row_major_data.ptr, input_vector.ptr, output_vector.ptr, m_rows, n_cols);
    } else {
        const m_rows_worker_thread = self.m_rows_worker_thread;
        const n_worker_thread_cols = m_rows_worker_thread * n_cols;

        for (self.worker_threads, 0..) |*worker_thread, worker_thread_index| {
            const worker_thread_row = worker_thread_index * n_worker_thread_cols;
            const output_vector_offset = worker_thread_index * m_rows_worker_thread;

            worker_thread.* = try std.Thread.spawn(.{}, matvecmul, .{
                row_major_data[worker_thread_row..][0..n_worker_thread_cols],
                input_vector,
                output_vector[output_vector_offset..][0..m_rows_worker_thread],
                n_cols,
            });
        }

        const main_thread_row = self.worker_threads.len * n_worker_thread_cols;

        matvecmul(
            row_major_data[main_thread_row..],
            input_vector,
            output_vector[(self.worker_threads.len * m_rows_worker_thread)..],
            n_cols,
        );

        for (self.worker_threads) |worker_thread| {
            worker_thread.join();
        }
    }
}

fn matvecmul(
    row_major_data: []const f32,
    input_vector: []const f32,
    output_vector: []f32,
    n_cols: usize,
) void {
    for (output_vector, 0..) |*element, row| {
        element.* = lib.dot(row_major_data[(row * n_cols)..][0..n_cols], input_vector);
    }
}
