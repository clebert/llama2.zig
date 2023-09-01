const Self = @This();

const std = @import("std");
const Cli = @import("./cli.zig");
const Matrix = @import("./matrix.zig");
const Vector = @import("./vector.zig");

allocator: std.mem.Allocator,
mmap: bool,
dim: usize,
hidden_dim: usize,
n_layers: usize,
n_heads: usize,
n_kv_heads: usize,
vocab_size: usize,
kv_dim: usize,
head_size: usize,
head_size_sqrt: f32,
n_groups: usize,

weights: struct {
    token_embedding_vector: Vector,

    attention_norm_vector: Vector,
    attention_queries_matrix: Matrix,
    attention_keys_matrix: Matrix,
    attention_values_matrix: Matrix,
    attention_output_matrix: Matrix,

    feed_forward_norm_vector: Vector,
    feed_forward_hidden_matrix: Matrix,
    feed_forward_output_matrix: Matrix,
    feed_forward_residual_matrix: Matrix,

    final_norm_vector: []const f32,
    classifier_matrix: Matrix,
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

    const signed_vocab_size: i32 = config_data[5];
    const dim: usize = @intCast(config_data[0]);
    const hidden_dim: usize = @intCast(config_data[1]);
    const n_layers: usize = @intCast(config_data[2]);
    const n_heads: usize = @intCast(config_data[3]);
    const n_kv_heads: usize = @intCast(config_data[4]);
    const vocab_size: usize = std.math.absCast(signed_vocab_size);
    const kv_dim: usize = (dim * n_kv_heads) / n_heads;
    const head_size: usize = dim / n_heads;
    const head_size_sqrt: f32 = std.math.sqrt(@as(f32, @floatFromInt(head_size)));
    const n_groups: usize = n_heads / n_kv_heads;

    var weights_data: [*]const f32 = @alignCast(@ptrCast(data[28..]));

    const token_embedding_vector = Vector.init(
        dim,
        readFloatSlice(&weights_data, vocab_size * dim),
    );

    const attention_norm_vector = Vector.init(
        dim,
        readFloatSlice(&weights_data, n_layers * dim),
    );

    const attention_queries_matrix = try Matrix.init(
        allocator,
        cli.multithreading,
        dim,
        dim,
        readFloatSlice(&weights_data, n_layers * (dim * dim)),
    );

    errdefer attention_queries_matrix.deinit();

    const attention_keys_matrix = try Matrix.init(
        allocator,
        cli.multithreading,
        kv_dim,
        dim,
        readFloatSlice(&weights_data, n_layers * (kv_dim * dim)),
    );

    errdefer attention_keys_matrix.deinit();

    const attention_values_matrix = try Matrix.init(
        allocator,
        cli.multithreading,
        kv_dim,
        dim,
        readFloatSlice(&weights_data, n_layers * (kv_dim * dim)),
    );

    errdefer attention_values_matrix.deinit();

    const attention_output_matrix = try Matrix.init(
        allocator,
        cli.multithreading,
        dim,
        dim,
        readFloatSlice(&weights_data, n_layers * (dim * dim)),
    );

    errdefer attention_output_matrix.deinit();

    const feed_forward_norm_vector = Vector.init(
        dim,
        readFloatSlice(&weights_data, n_layers * dim),
    );

    const feed_forward_hidden_matrix = try Matrix.init(
        allocator,
        cli.multithreading,
        hidden_dim,
        dim,
        readFloatSlice(&weights_data, n_layers * (hidden_dim * dim)),
    );

    errdefer feed_forward_hidden_matrix.deinit();

    const feed_forward_output_matrix = try Matrix.init(
        allocator,
        cli.multithreading,
        dim,
        hidden_dim,
        readFloatSlice(&weights_data, n_layers * (dim * hidden_dim)),
    );

    errdefer feed_forward_output_matrix.deinit();

    const feed_forward_residual_matrix = try Matrix.init(
        allocator,
        cli.multithreading,
        hidden_dim,
        dim,
        readFloatSlice(&weights_data, n_layers * (hidden_dim * dim)),
    );

    errdefer feed_forward_residual_matrix.deinit();

    const final_norm_vector = readFloatSlice(&weights_data, dim);
    const seq_len: usize = @intCast(config_data[6]);

    _ = readFloatSlice(&weights_data, seq_len * head_size / 2);
    _ = readFloatSlice(&weights_data, seq_len * head_size / 2);

    return Self{
        .allocator = allocator,
        .mmap = cli.mmap,
        .dim = dim,
        .hidden_dim = hidden_dim,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .vocab_size = vocab_size,
        .kv_dim = kv_dim,
        .head_size = head_size,
        .head_size_sqrt = head_size_sqrt,
        .n_groups = n_groups,

        .weights = .{
            .token_embedding_vector = token_embedding_vector,

            .attention_norm_vector = attention_norm_vector,
            .attention_queries_matrix = attention_queries_matrix,
            .attention_keys_matrix = attention_keys_matrix,
            .attention_values_matrix = attention_values_matrix,
            .attention_output_matrix = attention_output_matrix,

            .feed_forward_norm_vector = feed_forward_norm_vector,
            .feed_forward_hidden_matrix = feed_forward_hidden_matrix,
            .feed_forward_output_matrix = feed_forward_output_matrix,
            .feed_forward_residual_matrix = feed_forward_residual_matrix,

            .final_norm_vector = final_norm_vector,

            // https://github.com/karpathy/llama2.c/commit/c3e0d73bd294e1f5e4d17425fac09aaec536400d
            .classifier_matrix = try Matrix.init(
                allocator,
                cli.multithreading,
                vocab_size,
                dim,
                if (signed_vocab_size > 0)
                    token_embedding_vector.data
                else
                    readFloatSlice(&weights_data, vocab_size * dim),
            ),
        },

        .data = data,
    };
}

pub fn deinit(self: *const Self) void {
    self.weights.attention_queries_matrix.deinit();
    self.weights.attention_keys_matrix.deinit();
    self.weights.attention_values_matrix.deinit();
    self.weights.attention_output_matrix.deinit();
    self.weights.feed_forward_hidden_matrix.deinit();
    self.weights.feed_forward_output_matrix.deinit();
    self.weights.feed_forward_residual_matrix.deinit();
    self.weights.classifier_matrix.deinit();

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
