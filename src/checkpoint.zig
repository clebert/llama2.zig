const Self = @This();

const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const vector = @import("./vector.zig");

allocator: std.mem.Allocator,
embedding_size: usize,
ffn_hidden_size: usize,
n_layers: usize,
n_attention_heads: usize,
n_attention_query_groups: usize,
vocab_size: usize,
max_sequence_length: usize,
shared_output_matrix: bool,

weights: struct {
    token_embedding_vectors: Tensor(2),
    attention_norm_vectors: Tensor(2),
    attention_query_matrices: Tensor(3),
    attention_key_matrices: Tensor(3),
    attention_value_matrices: Tensor(3),
    attention_output_matrices: Tensor(3),
    ffn_norm_vectors: Tensor(2),
    ffn_gate_matrices: Tensor(3),
    ffn_down_matrices: Tensor(3),
    ffn_up_matrices: Tensor(3),
    output_norm_vector: Tensor(1),
    output_matrix: Tensor(2),
},

pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
    const v1_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ model_path, "checkpoint_v1.bin" },
    );

    defer allocator.free(v1_path);

    const v1_file = std.fs.cwd().openFile(v1_path, .{}) catch null;

    defer if (v1_file) |file| file.close();

    if (v1_file) |file| return try readV1(allocator, file);

    const legacy_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ model_path, "checkpoint_legacy.bin" },
    );

    defer allocator.free(legacy_path);

    const legacy_file = std.fs.cwd().openFile(legacy_path, .{}) catch null;

    defer if (legacy_file) |file| file.close();

    if (legacy_file) |file| return try readLegacy(allocator, file);

    return error.CheckpointFileNotFound;
}

// https://github.com/karpathy/llama2.c/blob/d9862069e7ef665fe6309e3c17398ded2f121bf5/export.py#L132
pub fn writeV1(self: *const Self, allocator: std.mem.Allocator, model_path: []const u8) !void {
    const path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ model_path, "checkpoint_v1.bin" },
    );

    defer allocator.free(path);

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });

    defer file.close();

    try file.writer().writeAll("ak42");
    try file.writer().writeIntLittle(i32, 1);
    try file.writer().writeIntLittle(i32, @as(i32, @intCast(self.embedding_size)));
    try file.writer().writeIntLittle(i32, @as(i32, @intCast(self.ffn_hidden_size)));
    try file.writer().writeIntLittle(i32, @as(i32, @intCast(self.n_layers)));
    try file.writer().writeIntLittle(i32, @as(i32, @intCast(self.n_attention_heads)));
    try file.writer().writeIntLittle(i32, @as(i32, @intCast(self.n_attention_query_groups)));
    try file.writer().writeIntLittle(i32, @as(i32, @intCast(self.vocab_size)));
    try file.writer().writeIntLittle(i32, @as(i32, @intCast(self.max_sequence_length)));
    try file.writer().writeIntLittle(u8, @as(u8, @intFromBool(self.shared_output_matrix)));
    try file.writer().writeByteNTimes(0, 256 - try file.getPos());
    try self.weights.attention_norm_vectors.write(file);
    try self.weights.ffn_norm_vectors.write(file);
    try self.weights.output_norm_vector.write(file);
    try self.weights.token_embedding_vectors.write(file);
    try self.weights.attention_query_matrices.write(file);
    try self.weights.attention_key_matrices.write(file);
    try self.weights.attention_value_matrices.write(file);
    try self.weights.attention_output_matrices.write(file);
    try self.weights.ffn_gate_matrices.write(file);
    try self.weights.ffn_down_matrices.write(file);
    try self.weights.ffn_up_matrices.write(file);

    if (!self.shared_output_matrix) {
        try self.weights.output_matrix.write(file);
    }
}

// https://github.com/karpathy/llama2.c/blob/d9862069e7ef665fe6309e3c17398ded2f121bf5/export.py#L132
fn readV1(allocator: std.mem.Allocator, file: std.fs.File) !Self {
    const magic: [*]const u8 = @ptrCast(&try file.reader().readIntLittle(u32));

    if (!std.mem.eql(u8, magic[0..4], "ak42")) {
        return error.InvalidMagic;
    }

    const version = try file.reader().readIntLittle(i32);

    if (version != 1) {
        return error.InvalidVersion;
    }

    const embedding_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const ffn_hidden_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_layers: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_attention_heads: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_attention_query_groups: usize = @intCast(try file.reader().readIntLittle(i32));
    const vocab_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const max_sequence_length: usize = @intCast(try file.reader().readIntLittle(i32));
    const shared_output_matrix = try file.reader().readIntLittle(u8) == 1;

    try file.seekTo(256);

    const attention_norm_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ n_layers, embedding_size },
    );

    errdefer attention_norm_vectors.deinit();
    try attention_norm_vectors.read(file);

    const ffn_norm_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ n_layers, embedding_size },
    );

    errdefer ffn_norm_vectors.deinit();
    try ffn_norm_vectors.read(file);

    const output_norm_vector = try Tensor(1).init(
        allocator,
        [_]usize{embedding_size},
    );

    errdefer output_norm_vector.deinit();
    try output_norm_vector.read(file);

    const token_embedding_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ vocab_size, embedding_size },
    );

    errdefer token_embedding_vectors.deinit();
    try token_embedding_vectors.read(file);

    const attention_query_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, embedding_size },
    );

    errdefer attention_query_matrices.deinit();
    try attention_query_matrices.read(file);

    const attention_head_size: usize = embedding_size / n_attention_heads;

    const attention_key_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, n_attention_query_groups * attention_head_size, embedding_size },
    );

    errdefer attention_key_matrices.deinit();
    try attention_key_matrices.read(file);

    const attention_value_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, n_attention_query_groups * attention_head_size, embedding_size },
    );

    errdefer attention_value_matrices.deinit();
    try attention_value_matrices.read(file);

    const attention_output_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, embedding_size },
    );

    errdefer attention_output_matrices.deinit();
    try attention_output_matrices.read(file);

    const ffn_gate_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, ffn_hidden_size, embedding_size },
    );

    errdefer ffn_gate_matrices.deinit();
    try ffn_gate_matrices.read(file);

    const ffn_down_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, ffn_hidden_size },
    );

    errdefer ffn_down_matrices.deinit();
    try ffn_down_matrices.read(file);

    const ffn_up_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, ffn_hidden_size, embedding_size },
    );

    errdefer ffn_up_matrices.deinit();
    try ffn_up_matrices.read(file);

    const output_matrix = if (shared_output_matrix)
        token_embedding_vectors
    else
        try Tensor(2).init(allocator, [_]usize{ vocab_size, embedding_size });

    errdefer if (!shared_output_matrix) {
        output_matrix.deinit();
    };

    if (!shared_output_matrix) {
        try output_matrix.read(file);
    }

    return Self{
        .allocator = allocator,
        .embedding_size = embedding_size,
        .ffn_hidden_size = ffn_hidden_size,
        .n_layers = n_layers,
        .n_attention_heads = n_attention_heads,
        .n_attention_query_groups = n_attention_query_groups,
        .vocab_size = vocab_size,
        .max_sequence_length = max_sequence_length,
        .shared_output_matrix = shared_output_matrix,

        .weights = .{
            .token_embedding_vectors = token_embedding_vectors,
            .attention_norm_vectors = attention_norm_vectors,
            .attention_query_matrices = attention_query_matrices,
            .attention_key_matrices = attention_key_matrices,
            .attention_value_matrices = attention_value_matrices,
            .attention_output_matrices = attention_output_matrices,
            .ffn_norm_vectors = ffn_norm_vectors,
            .ffn_gate_matrices = ffn_gate_matrices,
            .ffn_down_matrices = ffn_down_matrices,
            .ffn_up_matrices = ffn_up_matrices,
            .output_norm_vector = output_norm_vector,
            .output_matrix = output_matrix,
        },
    };
}

// https://github.com/karpathy/llama2.c/blob/d9862069e7ef665fe6309e3c17398ded2f121bf5/export.py#L75
fn readLegacy(allocator: std.mem.Allocator, file: std.fs.File) !Self {
    const embedding_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const ffn_hidden_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_layers: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_attention_heads: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_attention_query_groups: usize = @intCast(try file.reader().readIntLittle(i32));

    // https://github.com/karpathy/llama2.c/blob/35deb5e0fa55f0a257040bcf1624ed8386e63dc7/run.c#L153
    const signed_vocab_size = try file.reader().readIntLittle(i32);
    const shared_output_matrix = signed_vocab_size > 0;

    const vocab_size: usize = std.math.absCast(signed_vocab_size);
    const max_sequence_length: usize = @intCast(try file.reader().readIntLittle(i32));

    const token_embedding_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ vocab_size, embedding_size },
    );

    errdefer token_embedding_vectors.deinit();
    try token_embedding_vectors.read(file);

    const attention_norm_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ n_layers, embedding_size },
    );

    errdefer attention_norm_vectors.deinit();
    try attention_norm_vectors.read(file);

    const attention_query_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, embedding_size },
    );

    errdefer attention_query_matrices.deinit();
    try attention_query_matrices.read(file);

    const attention_head_size: usize = embedding_size / n_attention_heads;

    const attention_key_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, n_attention_query_groups * attention_head_size, embedding_size },
    );

    errdefer attention_key_matrices.deinit();
    try attention_key_matrices.read(file);

    const attention_value_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, n_attention_query_groups * attention_head_size, embedding_size },
    );

    errdefer attention_value_matrices.deinit();
    try attention_value_matrices.read(file);

    const attention_output_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, embedding_size },
    );

    errdefer attention_output_matrices.deinit();
    try attention_output_matrices.read(file);

    const ffn_norm_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ n_layers, embedding_size },
    );

    errdefer ffn_norm_vectors.deinit();
    try ffn_norm_vectors.read(file);

    const ffn_gate_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, ffn_hidden_size, embedding_size },
    );

    errdefer ffn_gate_matrices.deinit();
    try ffn_gate_matrices.read(file);

    const ffn_down_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, ffn_hidden_size },
    );

    errdefer ffn_down_matrices.deinit();
    try ffn_down_matrices.read(file);

    const ffn_up_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, ffn_hidden_size, embedding_size },
    );

    errdefer ffn_up_matrices.deinit();
    try ffn_up_matrices.read(file);

    const output_norm_vector = try Tensor(1).init(
        allocator,
        [_]usize{embedding_size},
    );

    errdefer output_norm_vector.deinit();
    try output_norm_vector.read(file);

    try file.seekBy(@intCast(max_sequence_length * attention_head_size * @sizeOf(f32)));

    const output_matrix = if (shared_output_matrix)
        token_embedding_vectors
    else
        try Tensor(2).init(allocator, [_]usize{ vocab_size, embedding_size });

    errdefer if (!shared_output_matrix) {
        output_matrix.deinit();
    };

    if (!shared_output_matrix) {
        try output_matrix.read(file);
    }

    return Self{
        .allocator = allocator,
        .embedding_size = embedding_size,
        .ffn_hidden_size = ffn_hidden_size,
        .n_layers = n_layers,
        .n_attention_heads = n_attention_heads,
        .n_attention_query_groups = n_attention_query_groups,
        .vocab_size = vocab_size,
        .max_sequence_length = max_sequence_length,
        .shared_output_matrix = shared_output_matrix,

        .weights = .{
            .token_embedding_vectors = token_embedding_vectors,
            .attention_norm_vectors = attention_norm_vectors,
            .attention_query_matrices = attention_query_matrices,
            .attention_key_matrices = attention_key_matrices,
            .attention_value_matrices = attention_value_matrices,
            .attention_output_matrices = attention_output_matrices,
            .ffn_norm_vectors = ffn_norm_vectors,
            .ffn_gate_matrices = ffn_gate_matrices,
            .ffn_down_matrices = ffn_down_matrices,
            .ffn_up_matrices = ffn_up_matrices,
            .output_norm_vector = output_norm_vector,
            .output_matrix = output_matrix,
        },
    };
}

pub fn deinit(self: *const Self) void {
    self.weights.token_embedding_vectors.deinit();
    self.weights.attention_norm_vectors.deinit();
    self.weights.attention_query_matrices.deinit();
    self.weights.attention_key_matrices.deinit();
    self.weights.attention_value_matrices.deinit();
    self.weights.attention_output_matrices.deinit();
    self.weights.ffn_norm_vectors.deinit();
    self.weights.ffn_gate_matrices.deinit();
    self.weights.ffn_down_matrices.deinit();
    self.weights.ffn_up_matrices.deinit();
    self.weights.output_norm_vector.deinit();

    if (!self.shared_output_matrix) {
        self.weights.output_matrix.deinit();
    }
}
