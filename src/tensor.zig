const std = @import("std");

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

        pub fn deinit(self: *const Self) void {
            if (self.allocator) |allocator| {
                allocator.free(self.values);
            }
        }

        pub fn read(self: *const Self, file: std.fs.File) !void {
            const values: [*]u8 = @ptrCast(self.values);

            try file.reader().readNoEof(values[0 .. self.values.len * @sizeOf(f32)]);
        }

        pub fn write(self: *const Self, file: std.fs.File) !void {
            const values: [*]u8 = @ptrCast(self.values);

            try file.writer().writeAll(values[0 .. self.values.len * @sizeOf(f32)]);
        }

        pub fn slice(self: *const Self, index: usize) Tensor(n_dims - 1) {
            comptime if (n_dims < 2) @compileError("n_dims < 2");

            const n_sub_values = @reduce(.Mul, @as(@Vector(n_dims - 1, usize), self.sub_dims));

            return .{
                .allocator = null,
                .sub_dims = self.sub_dims[1..].*,
                .values = self.values[index * n_sub_values ..][0..n_sub_values],
            };
        }

        pub fn add(self: *const Self, other: anytype) void {
            @setFloatMode(.Optimized);

            std.debug.assert(self.values.len == other.values.len);

            for (self.values, 0..) |*value, index| {
                value.* += other.values[index];
            }
        }

        pub fn computeMatrixVectorMultiplication(
            self: *const Self,
            input: anytype,
            output: anytype,
        ) void {
            for (output.values, 0..) |*value, index| {
                value.* = self.slice(index).computeScalarProduct(&input);
            }
        }

        pub fn computeScalarProduct(self: *const Self, other: anytype) f32 {
            if (self.values.len % 32 == 0) {
                return _computeScalarProduct(32, self, other);
            }

            if (self.values.len % 16 == 0) {
                return _computeScalarProduct(16, self, other);
            }

            if (self.values.len % 8 == 0) {
                return _computeScalarProduct(8, self, other);
            }

            return _computeScalarProduct(4, self, other);
        }

        // Pre-normalization using RMSNorm: https://arxiv.org/abs/1910.07467
        pub fn computeRMSNorm(self: *const Self, weight: anytype, output: anytype) void {
            @setFloatMode(.Optimized);

            std.debug.assert(output.values.len == self.values.len);
            std.debug.assert(output.values.len == weight.values.len);

            var rms_scaling_factor: f32 = 0;

            for (self.values) |value| {
                rms_scaling_factor += value * value;
            }

            rms_scaling_factor /= @floatFromInt(self.values.len);
            rms_scaling_factor += 1e-5;
            rms_scaling_factor = 1 / std.math.sqrt(rms_scaling_factor);

            for (output.values, 0..) |*value, index| {
                value.* = weight.values[index] * rms_scaling_factor * self.values[index];
            }
        }
    };
}

fn _computeScalarProduct(
    comptime vector_size: comptime_int,
    input_1: anytype,
    input_2: anytype,
) f32 {
    @setFloatMode(.Optimized);

    std.debug.assert(input_1.values.len == input_2.values.len);

    var output_values: @Vector(vector_size, f32) = @splat(0.0);
    var index: usize = 0;

    while (index < input_1.values.len) : (index += vector_size) {
        output_values +=
            @as(@Vector(vector_size, f32), input_1.values[index..][0..vector_size].*) *
            @as(@Vector(vector_size, f32), input_2.values[index..][0..vector_size].*);
    }

    return @reduce(.Add, output_values);
}
