const Self = @This();

const std = @import("std");
const Cli = @import("./cli.zig");
const MatrixArray = @import("./matrix_array.zig");
const VectorArray = @import("./vector_array.zig").VectorArray([]const f32);

allocator: std.mem.Allocator,

embedding_size: usize,
intermediate_size: usize,
n_layers: usize,
n_heads: usize,
n_query_groups: usize,
vocab_size: usize,
max_sequence_length: usize,

weights: struct {
    token_embedding_vectors: VectorArray,
    attention_norm_vectors: VectorArray,
    attention_query_projection_matrices: MatrixArray,
    attention_key_projection_matrices: MatrixArray,
    attention_value_projection_matrices: MatrixArray,
    attention_output_projection_matrices: MatrixArray,
    ffn_norm_vectors: VectorArray,
    ffn_hidden_projection_matrices: MatrixArray, // TODO: []Matrix
    ffn_output_projection_matrices: MatrixArray,
    ffn_scaling_projection_matrices: MatrixArray,
    final_norm_vector: []const f32,
    final_classifier_projection_matrices: MatrixArray, // TODO: only singular form is needed
},

data: []const u8,

// TODO: switch to file format v2
// TODO: write converter
pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const data = try readFile(allocator, cli.checkpoint_path);

    errdefer allocator.free(data);

    const config_data: [*]const i32 = @alignCast(@ptrCast(data[0..28]));

    const embedding_size: usize = @intCast(config_data[0]);
    const intermediate_size: usize = @intCast(config_data[1]);
    const n_layers: usize = @intCast(config_data[2]);
    const n_heads: usize = @intCast(config_data[3]);
    const n_query_groups: usize = @intCast(config_data[4]);
    const signed_vocab_size: i32 = config_data[5];
    const vocab_size: usize = std.math.absCast(signed_vocab_size);
    const max_sequence_length: usize = @intCast(config_data[6]);

    var weights_data: [*]const f32 = @alignCast(@ptrCast(data[28..]));

    const token_embedding_vectors = VectorArray.init(
        embedding_size,
        readFloatSlice(&weights_data, vocab_size * embedding_size),
    );

    const attention_norm_vectors = VectorArray.init(
        embedding_size,
        readFloatSlice(&weights_data, n_layers * embedding_size),
    );

    const attention_query_projection_matrices = MatrixArray.init(
        embedding_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (embedding_size * embedding_size)),
    );

    errdefer attention_query_projection_matrices.deinit();

    const head_size: usize = embedding_size / n_heads;
    const multi_head_key_value_size: usize = head_size * n_query_groups;

    const attention_key_projection_matrices = MatrixArray.init(
        multi_head_key_value_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (multi_head_key_value_size * embedding_size)),
    );

    errdefer attention_key_projection_matrices.deinit();

    const attention_value_projection_matrices = MatrixArray.init(
        multi_head_key_value_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (multi_head_key_value_size * embedding_size)),
    );

    errdefer attention_value_projection_matrices.deinit();

    const attention_output_projection_matrices = MatrixArray.init(
        embedding_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (embedding_size * embedding_size)),
    );

    errdefer attention_output_projection_matrices.deinit();

    const ffn_norm_vectors = VectorArray.init(
        embedding_size,
        readFloatSlice(&weights_data, n_layers * embedding_size),
    );

    const ffn_hidden_projection_matrices = MatrixArray.init(
        intermediate_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (intermediate_size * embedding_size)),
    );

    errdefer ffn_hidden_projection_matrices.deinit();

    const ffn_output_projection_matrices = MatrixArray.init(
        embedding_size,
        intermediate_size,
        readFloatSlice(&weights_data, n_layers * (embedding_size * intermediate_size)),
    );

    errdefer ffn_output_projection_matrices.deinit();

    const ffn_scaling_projection_matrices = MatrixArray.init(
        intermediate_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (intermediate_size * embedding_size)),
    );

    errdefer ffn_scaling_projection_matrices.deinit();

    const final_norm_vector = readFloatSlice(&weights_data, embedding_size);

    _ = readFloatSlice(&weights_data, max_sequence_length * head_size / 2);
    _ = readFloatSlice(&weights_data, max_sequence_length * head_size / 2);

    // https://github.com/karpathy/llama2.c/commit/c3e0d73bd294e1f5e4d17425fac09aaec536400d
    const final_classifier_projection_matrices = MatrixArray.init(
        vocab_size,
        embedding_size,
        if (signed_vocab_size > 0)
            token_embedding_vectors.data
        else
            readFloatSlice(&weights_data, vocab_size * embedding_size),
    );

    return Self{
        .allocator = allocator,

        .embedding_size = embedding_size,
        .intermediate_size = intermediate_size,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_query_groups = n_query_groups,
        .vocab_size = vocab_size,
        .max_sequence_length = max_sequence_length,

        .weights = .{
            .token_embedding_vectors = token_embedding_vectors,
            .attention_norm_vectors = attention_norm_vectors,
            .attention_query_projection_matrices = attention_query_projection_matrices,
            .attention_key_projection_matrices = attention_key_projection_matrices,
            .attention_value_projection_matrices = attention_value_projection_matrices,
            .attention_output_projection_matrices = attention_output_projection_matrices,
            .ffn_norm_vectors = ffn_norm_vectors,
            .ffn_hidden_projection_matrices = ffn_hidden_projection_matrices,
            .ffn_output_projection_matrices = ffn_output_projection_matrices,
            .ffn_scaling_projection_matrices = ffn_scaling_projection_matrices,
            .final_norm_vector = final_norm_vector,
            .final_classifier_projection_matrices = final_classifier_projection_matrices,
        },

        .data = data,
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.data);
}

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const stat = try file.stat();

    var data = try allocator.alloc(u8, stat.size);

    errdefer allocator.free(data);

    const n_bytes_read = try file.readAll(data);

    if (n_bytes_read != data.len) {
        return error.UnexpectedEndOfFile;
    }

    return data;
}

fn readFloatSlice(data: *[*]const f32, len: usize) []const f32 {
    const slice = data.*[0..len];

    data.* += len;

    return slice;
}
