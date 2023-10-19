const std = @import("std");

pub fn QuantizedTensor(comptime n_dims: comptime_int) type {
    comptime if (n_dims < 1) @compileError("n_dims < 1");

    return struct {
        const Self = @This();

        allocator: ?std.mem.Allocator,
        sub_dims: [n_dims - 1]usize,
        group_size: usize,
        values: []i8,
        scaling_factors: []f32,

        pub fn init(allocator: std.mem.Allocator, dims: [n_dims]usize, group_size: usize) !Self {
            const n_values = @reduce(.Mul, @as(@Vector(n_dims, usize), dims));

            if (n_values % group_size != 0) {
                return error.InvalidGroupSize;
            }

            const n_groups = n_values / group_size;

            return .{
                .allocator = allocator,
                .sub_dims = dims[1..].*,
                .group_size = group_size,
                .values = try allocator.alloc(i8, n_values),
                .scaling_factors = try allocator.alloc(f32, n_groups),
            };
        }

        pub fn deinit(self: *const Self) void {
            if (self.allocator) |allocator| {
                allocator.free(self.values);
                allocator.free(self.scaling_factors);
            }
        }

        pub fn slice(self: *const Self, index: usize) !QuantizedTensor(n_dims - 1) {
            comptime if (n_dims < 2) @compileError("n_dims < 2");

            const n_sub_values = @reduce(.Mul, @as(@Vector(n_dims - 1, usize), self.sub_dims));

            if (n_sub_values % self.group_size != 0) {
                return error.InvalidGroupSize;
            }

            const n_sub_groups = n_sub_values / self.group_size;

            return .{
                .allocator = null,
                .sub_dims = self.sub_dims[1..].*,
                .group_size = self.group_size,
                .values = self.values[index * n_sub_values ..][0..n_sub_values],
                .scaling_factors = self.scaling_factors[index * n_sub_groups ..][0..n_sub_groups],
            };
        }

        pub fn computeMatrixVectorMultiplication(
            self: *const Self,
            input: anytype,
            output: anytype,
        ) !void {
            for (output.values, 0..) |*value, index| {
                value.* = try (try self.slice(index)).computeScalarProduct(&input);
            }
        }

        fn computeScalarProduct(self: *const Self, other: anytype) !f32 {
            // https://github.com/karpathy/llama2.c/pull/312#issuecomment-1684140683
            if (self.group_size == 32) {
                return _computeScalarProduct(32, self, other);
            }

            if (self.group_size == 16) {
                return _computeScalarProduct(16, self, other);
            }

            if (self.group_size == 8) {
                return _computeScalarProduct(8, self, other);
            }

            if (self.group_size == 4) {
                return _computeScalarProduct(4, self, other);
            }

            return error.UnsupportedGroupSize;
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
    std.debug.assert(input_1.scaling_factors.len == input_2.scaling_factors.len);

    var output_value: f32 = 0;
    var index: usize = 0;

    while (index < input_1.values.len) : (index += vector_size) {
        const values: @Vector(vector_size, i32) =
            @as(@Vector(vector_size, i8), input_1.values[index..][0..vector_size].*) *
            @as(@Vector(vector_size, i8), input_2.values[index..][0..vector_size].*);

        output_value += @as(f32, @floatFromInt(@reduce(.Add, values))) *
            input_1.scaling_factors[index / vector_size] *
            input_2.scaling_factors[index / vector_size];
    }

    return output_value;
}
