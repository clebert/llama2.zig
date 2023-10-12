const std = @import("std");
const vector = @import("./vector.zig");

pub fn Tensor(comptime n_dims: comptime_int) type {
    comptime if (n_dims < 1) @compileError("n_dims < 1");

    return struct {
        const Self = @This();

        allocator: ?std.mem.Allocator,
        data: []f32,
        sub_dims: [n_dims - 1]usize,

        pub fn init(allocator: std.mem.Allocator, dims: [n_dims]usize) !Self {
            const data_size = @reduce(.Mul, @as(@Vector(n_dims, usize), dims));

            return .{
                .allocator = allocator,
                .data = try allocator.alloc(f32, data_size),
                .sub_dims = dims[1..].*,
            };
        }

        pub fn deinit(self: *const Self) void {
            if (self.allocator) |allocator| {
                allocator.free(self.data);
            }
        }

        pub fn read(self: *const Self, file: std.fs.File) !void {
            const buffer: [*]u8 = @ptrCast(self.data);
            const n_bytes = self.data.len * @sizeOf(f32);
            const n_bytes_read = try file.reader().readAll(buffer[0..n_bytes]);

            if (n_bytes_read != n_bytes) {
                return error.UnexpectedEndOfFile;
            }
        }

        pub fn slice(self: *const Self, index: usize) Tensor(n_dims - 1) {
            comptime if (n_dims < 2) @compileError("n_dims < 2");

            const sub_data_size = @reduce(.Mul, @as(@Vector(n_dims - 1, usize), self.sub_dims));

            return Tensor(n_dims - 1){
                .allocator = null,
                .data = self.data[index * sub_data_size ..][0..sub_data_size],
                .sub_dims = self.sub_dims[1..].*,
            };
        }

        pub fn multiplyVector(self: *const Self, input: []const f32, output: []f32) void {
            comptime if (n_dims < 2) @compileError("n_dims < 2");

            const sub_data_size = @reduce(.Mul, @as(@Vector(n_dims - 1, usize), self.sub_dims));

            std.debug.assert(input.len == sub_data_size);
            std.debug.assert(output.len == self.data.len / sub_data_size);

            for (output, 0..) |*value, index| {
                value.* = vector.dot(self.data[index * sub_data_size ..][0..sub_data_size], input);
            }
        }
    };
}
