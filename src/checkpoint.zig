const Self = @This();

const std = @import("std");
const Cli = @import("./cli.zig");
const Tensor = @import("./tensor.zig").Tensor;
const vector = @import("./vector.zig");

allocator: std.mem.Allocator,
embedding_size: usize,
hidden_size: usize,
n_layers: usize,
n_heads: usize,
n_query_groups: usize,
vocab_size: usize,
max_sequence_length: usize,
shared_final_classifier_matrix: bool,

weights: struct {
    token_embedding_vectors: Tensor(2),
    attention_pre_norm_vectors: Tensor(2),
    attention_query_matrices: Tensor(3),
    attention_key_matrices: Tensor(3),
    attention_value_matrices: Tensor(3),
    attention_output_matrices: Tensor(3),
    ffn_pre_norm_vectors: Tensor(2),
    ffn_pre_activation_matrices: Tensor(3),
    ffn_output_matrices: Tensor(3),
    ffn_gate_matrices: Tensor(3),
    final_norm_vector: Tensor(1),
    final_classifier_matrix: Tensor(2),
},

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const file = try std.fs.cwd().openFile(cli.checkpoint_path, .{});

    defer file.close();

    const embedding_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const hidden_size: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_layers: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_heads: usize = @intCast(try file.reader().readIntLittle(i32));
    const n_query_groups: usize = @intCast(try file.reader().readIntLittle(i32));

    // https://github.com/karpathy/llama2.c/blob/35deb5e0fa55f0a257040bcf1624ed8386e63dc7/run.c#L153
    const signed_vocab_size = try file.reader().readIntLittle(i32);
    const shared_final_classifier_matrix = signed_vocab_size > 0;

    const vocab_size: usize = std.math.absCast(signed_vocab_size);
    const max_sequence_length: usize = @intCast(try file.reader().readIntLittle(i32));

    const token_embedding_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ vocab_size, embedding_size },
    );

    errdefer token_embedding_vectors.deinit();
    try token_embedding_vectors.read(file);

    const attention_pre_norm_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ n_layers, embedding_size },
    );

    errdefer attention_pre_norm_vectors.deinit();
    try attention_pre_norm_vectors.read(file);

    const attention_query_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, embedding_size },
    );

    errdefer attention_query_matrices.deinit();
    try attention_query_matrices.read(file);

    const head_size: usize = embedding_size / n_heads;

    const attention_key_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, n_query_groups * head_size, embedding_size },
    );

    errdefer attention_key_matrices.deinit();
    try attention_key_matrices.read(file);

    const attention_value_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, n_query_groups * head_size, embedding_size },
    );

    errdefer attention_value_matrices.deinit();
    try attention_value_matrices.read(file);

    const attention_output_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, embedding_size },
    );

    errdefer attention_output_matrices.deinit();
    try attention_output_matrices.read(file);

    const ffn_pre_norm_vectors = try Tensor(2).init(
        allocator,
        [_]usize{ n_layers, embedding_size },
    );

    errdefer ffn_pre_norm_vectors.deinit();
    try ffn_pre_norm_vectors.read(file);

    const ffn_pre_activation_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, hidden_size, embedding_size },
    );

    errdefer ffn_pre_activation_matrices.deinit();
    try ffn_pre_activation_matrices.read(file);

    const ffn_output_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, embedding_size, hidden_size },
    );

    errdefer ffn_output_matrices.deinit();
    try ffn_output_matrices.read(file);

    const ffn_gate_matrices = try Tensor(3).init(
        allocator,
        [_]usize{ n_layers, hidden_size, embedding_size },
    );

    errdefer ffn_gate_matrices.deinit();
    try ffn_gate_matrices.read(file);

    const final_norm_vector = try Tensor(1).init(allocator, [_]usize{embedding_size});

    errdefer final_norm_vector.deinit();
    try final_norm_vector.read(file);

    try file.seekBy(@intCast(max_sequence_length * head_size * @sizeOf(f32)));

    const final_classifier_matrix = if (shared_final_classifier_matrix)
        token_embedding_vectors
    else
        try Tensor(2).init(allocator, [_]usize{ vocab_size, embedding_size });

    errdefer if (!shared_final_classifier_matrix) {
        final_classifier_matrix.deinit();
    };

    if (!shared_final_classifier_matrix) {
        try final_classifier_matrix.read(file);
    }

    return Self{
        .allocator = allocator,
        .embedding_size = embedding_size,
        .hidden_size = hidden_size,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_query_groups = n_query_groups,
        .vocab_size = vocab_size,
        .max_sequence_length = max_sequence_length,
        .shared_final_classifier_matrix = shared_final_classifier_matrix,

        .weights = .{
            .token_embedding_vectors = token_embedding_vectors,
            .attention_pre_norm_vectors = attention_pre_norm_vectors,
            .attention_query_matrices = attention_query_matrices,
            .attention_key_matrices = attention_key_matrices,
            .attention_value_matrices = attention_value_matrices,
            .attention_output_matrices = attention_output_matrices,
            .ffn_pre_norm_vectors = ffn_pre_norm_vectors,
            .ffn_pre_activation_matrices = ffn_pre_activation_matrices,
            .ffn_output_matrices = ffn_output_matrices,
            .ffn_gate_matrices = ffn_gate_matrices,
            .final_norm_vector = final_norm_vector,
            .final_classifier_matrix = final_classifier_matrix,
        },
    };
}

pub fn deinit(self: *const Self) void {
    self.weights.token_embedding_vectors.deinit();
    self.weights.attention_pre_norm_vectors.deinit();
    self.weights.attention_query_matrices.deinit();
    self.weights.attention_key_matrices.deinit();
    self.weights.attention_value_matrices.deinit();
    self.weights.attention_output_matrices.deinit();
    self.weights.ffn_pre_norm_vectors.deinit();
    self.weights.ffn_pre_activation_matrices.deinit();
    self.weights.ffn_output_matrices.deinit();
    self.weights.ffn_gate_matrices.deinit();
    self.weights.final_norm_vector.deinit();

    if (!self.shared_final_classifier_matrix) {
        self.weights.final_classifier_matrix.deinit();
    }
}
