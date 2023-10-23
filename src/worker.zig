const Self = @This();

const std = @import("std");
const Vector = @import("vector.zig");

computation: ?MatrixVectorMultiplication = null,
mutex: std.Thread.Mutex = .{},
condition: std.Thread.Condition = .{},

pub fn spawn(self: *Self) !void {
    const thread = try std.Thread.spawn(.{}, compute, .{self});

    thread.detach();
}

pub fn schedule(self: *Self, computation: MatrixVectorMultiplication) void {
    self.mutex.lock();

    defer self.mutex.unlock();

    while (self.computation != null) {
        self.condition.wait(&self.mutex);
    }

    self.computation = computation;

    self.condition.signal();
}

pub fn wait(self: *Self) void {
    self.mutex.lock();

    defer self.mutex.unlock();

    while (self.computation != null) {
        self.condition.wait(&self.mutex);
    }
}

fn compute(self: *Self) !void {
    while (true) {
        self.mutex.lock();

        defer self.mutex.unlock();

        if (self.computation) |computation| {
            try computation.run();

            self.computation = null;

            self.condition.signal();

            continue;
        }

        self.condition.wait(&self.mutex);
    }
}

pub const MatrixVectorMultiplication = struct {
    rows: []const Vector,
    input: Vector,
    output_data: []f32,

    pub fn run(self: MatrixVectorMultiplication) !void {
        std.debug.assert(self.rows.len == self.output_data.len);

        for (self.output_data, 0..) |*element, index| {
            element.* = try self.rows[index].computeScalarProduct(self.input);
        }
    }
};
