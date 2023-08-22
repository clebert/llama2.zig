const linear_algebra = @import("lib/linear_algebra.zig");

pub const add = linear_algebra.add;
pub const dotProduct = linear_algebra.dotProduct;
pub const matmul = linear_algebra.matmul;
pub const matmul2 = linear_algebra.matmul2;
pub const matmul3 = linear_algebra.matmul3;
pub const rmsnorm = @import("lib/rmsnorm.zig").rmsnorm;
pub const rope = @import("lib/rope.zig").rope;
