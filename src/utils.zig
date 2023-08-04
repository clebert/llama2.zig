const std = @import("std");

pub fn argmax(v: []f32) usize {
    // return argmax of v in elements 0..n
    var max_i: usize = 0;
    var max_p: f32 = v[0];

    for (1..v.len) |i| {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }

    return max_i;
}

pub fn sample(rng: *std.rand.DefaultPrng, probabilities: []f32) usize {
    var r = rng.random().float(f32);
    var cdf: f32 = 0.0;

    for (probabilities, 0..) |probability, i| {
        cdf += probability;

        if (r < cdf) {
            return i;
        }
    }

    return probabilities.len - 1;
}

pub fn matmul(xout: []f32, x: []const f32, w: []const f32) void {
    @setFloatMode(.Optimized);

    const v_len: comptime_int = 32;

    std.debug.assert(w.len >= xout.len * x.len);
    std.debug.assert(x.len % v_len == 0);

    for (xout, 0..) |*xoutptr, i| {
        var value: f32 = 0;

        const i_n = i * x.len;
        var j: usize = 0;

        // https://github.com/karpathy/llama2.c/pull/95
        while (j < x.len) : (j += v_len) {
            value += @reduce(
                .Add,
                @as(@Vector(v_len, f32), w[(i_n + j)..][0..v_len].*) * @as(@Vector(v_len, f32), x[j..][0..v_len].*),
            );
        }

        xoutptr.* = value;
    }
}

pub fn softmax(x: []f32) void {
    var max_val = std.mem.max(f32, x);

    // exp and sum
    var sum: f32 = 0.0;

    for (x) |*item| {
        item.* = std.math.exp(item.* - max_val);

        sum += item.*;
    }

    // normalize
    for (x) |*item| {
        item.* /= sum;
    }
}

test "compute softmax" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0 };

    const expected = [_]f32{
        0.02364052406, 0.06426167518, 0.1746813036, 0.474833058,
        0.02364052406, 0.06426167518, 0.1746813036,
    };

    softmax(x[0..]); // TODO: sum should be 1

    for (x, 0..) |item, i| {
        try std.testing.expectApproxEqAbs(expected[i], item, 0.00001);
    }
}

pub fn rmsnorm(o: []f32, x: []const f32, weight: []const f32) void {
    std.debug.assert(o.len == x.len);
    std.debug.assert(weight.len >= o.len);

    // calculate sum of squares
    var ss: f32 = 0.0;
    for (x) |item| {
        ss += item * item;
    }
    ss /= @floatFromInt(x.len);
    ss += 1e-5;
    ss = 1.0 / std.math.sqrt(ss);

    // normalize and scale
    for (o, 0..) |*item, i| {
        item.* = weight[i] * (ss * x[i]);
    }
}

test "rms normalization" {
    var o = [_]f32{ 0, 0 };
    const x = [_]f32{ 2, 3 };
    const weight = [_]f32{ 0.5, 0.5 };

    rmsnorm(o[0..], x[0..], weight[0..]);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5) * ((1.0 / std.math.sqrt(13.0 / 2.0 + 1e-5)) * 2), o[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5) * ((1.0 / std.math.sqrt(13.0 / 2.0 + 1e-5)) * 3), o[1], 1e-5);
}

pub fn accum(a: []f32, b: []const f32) void {
    std.debug.assert(a.len == b.len);

    for (a, 0..) |*item, index| {
        item.* += b[index];
    }
}

test "accumulate two slices" {
    var a = [_]f32{ 42, 85 };
    const b = [_]f32{ 100, 200 };
    const expected = [_]f32{ 142, 285 };

    accum(a[0..], b[0..]);

    try std.testing.expectEqualSlices(f32, expected[0..], a[0..]);
}