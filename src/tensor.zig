const std = @import("std");
const vector = @import("./vector.zig");

pub fn Tensor(comptime n_dims: comptime_int) type {
    comptime if (n_dims < 1) @compileError("TODO");

    return struct {
        const Self = @This();

        view: bool,
        allocator: ?std.mem.Allocator,
        data: []f32,
        sub_tensor_sizes: []const usize,

        pub fn init(allocator: std.mem.Allocator, dims: [n_dims]usize) !Self {
            var tensor_size: usize = 1;

            for (dims) |dim| tensor_size *= dim;

            const sub_tensor_sizes = try allocator.alloc(usize, n_dims - 1);

            for (sub_tensor_sizes, 1..) |*sub_tensor_size, dims_offset| {
                sub_tensor_size.* = 1;

                for (dims[dims_offset..]) |dim| sub_tensor_size.* *= dim;
            }

            return .{
                .view = false,
                .allocator = allocator,
                .data = try allocator.alignedAlloc(f32, std.atomic.cache_line, tensor_size),
                .sub_tensor_sizes = sub_tensor_sizes,
            };
        }

        pub fn initView(allocator: std.mem.Allocator, data: []f32, dims: [n_dims]usize) !Self {
            const sub_tensor_sizes = try allocator.alloc(usize, n_dims - 1);

            for (sub_tensor_sizes, 1..) |*sub_tensor_size, dims_offset| {
                sub_tensor_size.* = 1;

                for (dims[dims_offset..]) |dim| sub_tensor_size.* *= dim;
            }

            return .{
                .view = true,
                .allocator = allocator,
                .data = data,
                .sub_tensor_sizes = sub_tensor_sizes,
            };
        }

        pub fn deinit(self: *const Self) void {
            if (self.allocator) |allocator| {
                if (!self.view) {
                    allocator.free(@as([]align(std.atomic.cache_line) f32, @alignCast(self.data)));
                }

                allocator.free(self.sub_tensor_sizes);
            }
        }

        pub fn slice(self: *const Self, index: usize) Tensor(n_dims - 1) {
            comptime if (n_dims < 2) @compileError("TODO");

            const sub_tensor_size = self.sub_tensor_sizes[0];

            return Tensor(n_dims - 1){
                .view = self.view,
                .allocator = null,
                .data = self.data[(index * sub_tensor_size)..][0..sub_tensor_size],
                .sub_tensor_sizes = self.sub_tensor_sizes[1..],
            };
        }

        pub fn multiplyVector(self: *const Self, input_data: []const f32, output_data: []f32) void {
            comptime if (n_dims < 2) @compileError("TODO");

            const data = self.data;
            const sub_tensor_size = self.sub_tensor_sizes[0];

            std.debug.assert(input_data.len == sub_tensor_size);
            std.debug.assert(output_data.len == data.len / sub_tensor_size);

            for (output_data, 0..) |*value, index| {
                value.* = vector.dot(
                    data[(index * sub_tensor_size)..][0..sub_tensor_size],
                    input_data,
                );
            }
        }
    };
}
