const Self = @This();

const std = @import("std");
const Matrix = @import("matrix.zig");
const Vector = @import("vector.zig");

embedding_size: usize,
ffn_hidden_size: usize,
n_layers: usize,
n_attention_heads: usize,
n_attention_query_groups: usize,
vocab_size: usize,
max_sequence_length: usize,
embedding_weights: []const Vector,
attention_norm_weights: []const Vector,
attention_query_weights: []const Matrix,
attention_key_weights: []const Matrix,
attention_value_weights: []const Matrix,
attention_output_weights: []const Matrix,
ffn_norm_weights: []const Vector,
ffn_gate_weights: []const Matrix,
ffn_down_weights: []const Matrix,
ffn_up_weights: []const Matrix,
output_norm_weight: Vector,
output_weight: Matrix,

pub fn initLeaky(allocator: std.mem.Allocator, args: anytype) !Self {
    const path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ args.model_path, "checkpoint_v1.bin" },
    );

    defer allocator.free(path);

    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    if (try file.reader().readIntLittle(u32) != 0x616b3432) return error.InvalidMagic;
    if (try file.reader().readIntLittle(i32) != 1) return error.InvalidVersion;

    const embedding_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const ffn_hidden_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_layers: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_attention_heads: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_attention_query_groups: usize = @intCast(try file.reader().readIntLittle(i32));
    const vocab_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const max_sequence_length: usize = @intCast(try file.reader().readIntLittle(i32));
    const shared_output_weight = try file.reader().readIntLittle(u8) == 1;

    try file.seekTo(256);

    const attention_norm_weights = try Vector.initAllLeaky(allocator, n_layers, embedding_size);

    try Vector.readAll(file, attention_norm_weights);

    const ffn_norm_weights = try Vector.initAllLeaky(allocator, n_layers, embedding_size);

    try Vector.readAll(file, ffn_norm_weights);

    const output_norm_weight = try Vector.initLeaky(allocator, embedding_size);

    try output_norm_weight.read(file);

    const embedding_weights = try Vector.initAllLeaky(allocator, vocab_size, embedding_size);

    try Vector.readAll(file, embedding_weights);

    const attention_query_weights = try Matrix.initAllLeaky(
        allocator,
        n_layers,
        embedding_size,
        embedding_size,
    );

    try Matrix.readAll(file, attention_query_weights);

    const attention_head_size: usize = embedding_size / n_attention_heads;

    const attention_key_weights = try Matrix.initAllLeaky(
        allocator,
        n_layers,
        n_attention_query_groups * attention_head_size,
        embedding_size,
    );

    try Matrix.readAll(file, attention_key_weights);

    const attention_value_weights = try Matrix.initAllLeaky(
        allocator,
        n_layers,
        n_attention_query_groups * attention_head_size,
        embedding_size,
    );

    try Matrix.readAll(file, attention_value_weights);

    const attention_output_weights = try Matrix.initAllLeaky(
        allocator,
        n_layers,
        embedding_size,
        embedding_size,
    );

    try Matrix.readAll(file, attention_output_weights);

    const ffn_gate_weights = try Matrix.initAllLeaky(
        allocator,
        n_layers,
        ffn_hidden_size,
        embedding_size,
    );

    try Matrix.readAll(file, ffn_gate_weights);

    const ffn_down_weights = try Matrix.initAllLeaky(
        allocator,
        n_layers,
        embedding_size,
        ffn_hidden_size,
    );

    try Matrix.readAll(file, ffn_down_weights);

    const ffn_up_weights = try Matrix.initAllLeaky(
        allocator,
        n_layers,
        ffn_hidden_size,
        embedding_size,
    );

    try Matrix.readAll(file, ffn_up_weights);

    const output_weight = if (!shared_output_weight)
        try Matrix.initLeaky(allocator, vocab_size, embedding_size)
    else
        Matrix{ .rows = embedding_weights };

    if (!shared_output_weight) try output_weight.read(file);

    return .{
        .embedding_size = embedding_size,
        .ffn_hidden_size = ffn_hidden_size,
        .n_layers = n_layers,
        .n_attention_heads = n_attention_heads,
        .n_attention_query_groups = n_attention_query_groups,
        .vocab_size = vocab_size,
        .max_sequence_length = max_sequence_length,
        .embedding_weights = embedding_weights,
        .attention_norm_weights = attention_norm_weights,
        .attention_query_weights = attention_query_weights,
        .attention_key_weights = attention_key_weights,
        .attention_value_weights = attention_value_weights,
        .attention_output_weights = attention_output_weights,
        .ffn_norm_weights = ffn_norm_weights,
        .ffn_gate_weights = ffn_gate_weights,
        .ffn_down_weights = ffn_down_weights,
        .ffn_up_weights = ffn_up_weights,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
    };
}
