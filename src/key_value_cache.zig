const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Checkpoint = @import("checkpoint.zig");
const VectorArray = @import("./vector_array.zig").VectorArray([]f32);

allocator: std.mem.Allocator,

n_groups: usize,
head_size: usize,
sequence_length: usize,

data: []f32,

pub fn init(
    allocator: std.mem.Allocator,
    n_layers: usize,
    n_groups: usize,
    head_size: usize,
    sequence_length: usize,
) !Self {
    return Self{
        .allocator = allocator,

        .n_groups = n_groups,
        .head_size = head_size,
        .sequence_length = sequence_length,

        .data = try allocator.alloc(f32, n_layers * sequence_length * n_groups * head_size),
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.data);
}

pub fn at(self: *const Self, layer: usize, optional_position: ?usize) VectorArray {
    const total_group_size = self.n_groups * self.head_size;
    const data_len = self.sequence_length * total_group_size;
    const data = VectorArray.init(total_group_size, self.data[(layer * data_len)..][0..data_len]);

    if (optional_position) |position| {
        return VectorArray.init(self.head_size, data.at(position));
    }

    return data;
}
