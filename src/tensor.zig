const std = @import("std");
const simd = @import("simd.zig");

pub fn Tensor(comptime n_dims: comptime_int) type {
    comptime if (n_dims < 1) @compileError("n_dims < 1");

    return struct {
        const Self = @This();

        allocator: ?std.mem.Allocator,
        sub_dims: [n_dims - 1]usize,
        values: []f32,

        pub fn init(allocator: std.mem.Allocator, dims: [n_dims]usize) !Self {
            const n_values = @reduce(.Mul, @as(@Vector(n_dims, usize), dims));

            return .{
                .allocator = allocator,
                .sub_dims = dims[1..].*,
                .values = try allocator.alloc(f32, n_values),
            };
        }

        pub fn deinit(self: Self) void {
            if (self.allocator) |allocator| {
                allocator.free(self.values);
            }
        }

        pub fn read(self: Self, file: std.fs.File) !void {
            const values: [*]u8 = @ptrCast(self.values);

            try file.reader().readNoEof(values[0 .. self.values.len * @sizeOf(f32)]);
        }

        pub fn write(self: Self, file: std.fs.File) !void {
            const values: [*]u8 = @ptrCast(self.values);

            try file.writer().writeAll(values[0 .. self.values.len * @sizeOf(f32)]);
        }

        pub fn slice(self: Self, index: usize) Tensor(n_dims - 1) {
            comptime if (n_dims < 2) @compileError("n_dims < 2");

            const n_sub_values = @reduce(.Mul, @as(@Vector(n_dims - 1, usize), self.sub_dims));

            return .{
                .allocator = null,
                .sub_dims = self.sub_dims[1..].*,
                .values = self.values[index * n_sub_values ..][0..n_sub_values],
            };
        }

        pub fn computeMatrixVectorMultiplication(self: Self, input: anytype, output: anytype) void {
            for (output.values, 0..) |*value, index| {
                value.* = self.slice(index).computeScalarProduct(input);
            }
        }

        pub fn computeRMSNorm(self: Self, weight: anytype, output: anytype) void {
            if (self.values.len % 32 == 0)
                simd.computeRMSNorm(f32, 32, self.values, weight.values, output.values)
            else if (self.values.len % 16 == 0)
                simd.computeRMSNorm(f32, 16, self.values, weight.values, output.values)
            else if (self.values.len % 8 == 0)
                simd.computeRMSNorm(f32, 8, self.values, weight.values, output.values)
            else
                simd.computeRMSNorm(f32, 4, self.values, weight.values, output.values);
        }

        pub fn computeScalarProduct(self: Self, other: anytype) f32 {
            return if (self.values.len % 32 == 0)
                simd.computeScalarProduct(f32, 32, self.values, other.values)
            else if (self.values.len % 16 == 0)
                simd.computeScalarProduct(f32, 16, self.values, other.values)
            else if (self.values.len % 8 == 0)
                simd.computeScalarProduct(f32, 8, self.values, other.values)
            else
                simd.computeScalarProduct(f32, 4, self.values, other.values);
        }

        pub fn computeVectorAddition(self: Self, other: anytype) void {
            if (self.values.len % 32 == 0)
                simd.computeVectorAddition(f32, 32, self.values, other.values, self.values)
            else if (self.values.len % 16 == 0)
                simd.computeVectorAddition(f32, 16, self.values, other.values, self.values)
            else if (self.values.len % 8 == 0)
                simd.computeVectorAddition(f32, 8, self.values, other.values, self.values)
            else
                simd.computeVectorAddition(f32, 4, self.values, other.values, self.values);
        }
    };
}
