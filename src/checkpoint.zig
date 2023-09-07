const Self = @This();

const std = @import("std");
const Cli = @import("./cli.zig");
const MatrixArray = @import("./matrix_array.zig");
const VectorArray = @import("./vector_array.zig").VectorArray([]const f32);

allocator: std.mem.Allocator,
mmap: bool,

embedding_size: usize,
intermediate_size: usize,
n_layers: usize,
n_heads: usize,
n_query_groups: usize,
vocab_size: usize,
max_sequence_length: usize,

weights: struct {
    embedding_vectors: VectorArray,
    attention_norm_vectors: VectorArray,
    attention_query_matrices: MatrixArray,
    attention_key_matrices: MatrixArray,
    attention_value_matrices: MatrixArray,
    attention_output_matrices: MatrixArray,
    feed_forward_norm_vectors: VectorArray,
    feed_forward_hidden_matrices: MatrixArray,
    feed_forward_output_matrices: MatrixArray,
    feed_forward_scaling_matrices: MatrixArray,
    final_norm_vector: []const f32,
    classifier_matrices: MatrixArray,
},

data: []align(std.mem.page_size) const u8,

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const data = try if (cli.mmap)
        mmapFile(cli.checkpoint_path)
    else
        readFile(allocator, cli.checkpoint_path);

    errdefer if (cli.mmap) {
        std.os.munmap(data);
    } else {
        allocator.free(data);
    };

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

    const embedding_vectors = VectorArray.init(
        embedding_size,
        readFloatSlice(&weights_data, vocab_size * embedding_size),
    );

    const attention_norm_vectors = VectorArray.init(
        embedding_size,
        readFloatSlice(&weights_data, n_layers * embedding_size),
    );

    const attention_query_matrices = try MatrixArray.init(
        allocator,
        embedding_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (embedding_size * embedding_size)),
        cli.multithreading,
    );

    errdefer attention_query_matrices.deinit();

    const head_size: usize = embedding_size / n_heads;
    const multi_head_key_value_size: usize = head_size * n_query_groups;

    const attention_key_matrices = try MatrixArray.init(
        allocator,
        multi_head_key_value_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (multi_head_key_value_size * embedding_size)),
        cli.multithreading,
    );

    errdefer attention_key_matrices.deinit();

    const attention_value_matrices = try MatrixArray.init(
        allocator,
        multi_head_key_value_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (multi_head_key_value_size * embedding_size)),
        cli.multithreading,
    );

    errdefer attention_value_matrices.deinit();

    const attention_output_matrices = try MatrixArray.init(
        allocator,
        embedding_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (embedding_size * embedding_size)),
        cli.multithreading,
    );

    errdefer attention_output_matrices.deinit();

    const feed_forward_norm_vectors = VectorArray.init(
        embedding_size,
        readFloatSlice(&weights_data, n_layers * embedding_size),
    );

    const feed_forward_hidden_matrices = try MatrixArray.init(
        allocator,
        intermediate_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (intermediate_size * embedding_size)),
        cli.multithreading,
    );

    errdefer feed_forward_hidden_matrices.deinit();

    const feed_forward_output_matrices = try MatrixArray.init(
        allocator,
        embedding_size,
        intermediate_size,
        readFloatSlice(&weights_data, n_layers * (embedding_size * intermediate_size)),
        cli.multithreading,
    );

    errdefer feed_forward_output_matrices.deinit();

    const feed_forward_scaling_matrices = try MatrixArray.init(
        allocator,
        intermediate_size,
        embedding_size,
        readFloatSlice(&weights_data, n_layers * (intermediate_size * embedding_size)),
        cli.multithreading,
    );

    errdefer feed_forward_scaling_matrices.deinit();

    const final_norm_vector = readFloatSlice(&weights_data, embedding_size);

    _ = readFloatSlice(&weights_data, max_sequence_length * head_size / 2);
    _ = readFloatSlice(&weights_data, max_sequence_length * head_size / 2);

    // https://github.com/karpathy/llama2.c/commit/c3e0d73bd294e1f5e4d17425fac09aaec536400d
    const classifier_matrices = try MatrixArray.init(
        allocator,
        vocab_size,
        embedding_size,
        if (signed_vocab_size > 0)
            embedding_vectors.data
        else
            readFloatSlice(&weights_data, vocab_size * embedding_size),
        cli.multithreading,
    );

    return Self{
        .allocator = allocator,
        .mmap = cli.mmap,

        .embedding_size = embedding_size,
        .intermediate_size = intermediate_size,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_query_groups = n_query_groups,
        .vocab_size = vocab_size,
        .max_sequence_length = max_sequence_length,

        .weights = .{
            .embedding_vectors = embedding_vectors,
            .attention_norm_vectors = attention_norm_vectors,
            .attention_query_matrices = attention_query_matrices,
            .attention_key_matrices = attention_key_matrices,
            .attention_value_matrices = attention_value_matrices,
            .attention_output_matrices = attention_output_matrices,
            .feed_forward_norm_vectors = feed_forward_norm_vectors,
            .feed_forward_hidden_matrices = feed_forward_hidden_matrices,
            .feed_forward_output_matrices = feed_forward_output_matrices,
            .feed_forward_scaling_matrices = feed_forward_scaling_matrices,
            .final_norm_vector = final_norm_vector,
            .classifier_matrices = classifier_matrices,
        },

        .data = data,
    };
}

pub fn deinit(self: *const Self) void {
    self.weights.attention_query_matrices.deinit();
    self.weights.attention_key_matrices.deinit();
    self.weights.attention_value_matrices.deinit();
    self.weights.attention_output_matrices.deinit();
    self.weights.feed_forward_hidden_matrices.deinit();
    self.weights.feed_forward_output_matrices.deinit();
    self.weights.feed_forward_scaling_matrices.deinit();
    self.weights.classifier_matrices.deinit();

    if (self.mmap) {
        std.os.munmap(self.data);
    } else {
        self.allocator.free(self.data);
    }
}

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]align(std.mem.page_size) const u8 {
    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const stat = try file.stat();

    var data = try allocator.alignedAlloc(u8, std.mem.page_size, stat.size);

    errdefer allocator.free(data);

    const n_bytes_read = try file.readAll(data);

    if (n_bytes_read != data.len) {
        return error.UnexpectedEndOfFile;
    }

    return data;
}

fn mmapFile(path: []const u8) ![]align(std.mem.page_size) const u8 {
    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const stat = try file.stat();

    return std.os.mmap(
        null,
        stat.size,
        std.os.PROT.READ,
        std.os.MAP.PRIVATE,
        file.handle,
        0,
    );
}

fn readFloatSlice(data: *[*]const f32, len: usize) []const f32 {
    const slice = data.*[0..len];

    data.* += len;

    return slice;
}
