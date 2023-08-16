const std = @import("std");

pub const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
};

pub const Weights = struct {
    token_embedding_table: []f32, // vocab_size * dim
    rms_att_weight: []f32, // n_layers * dim
    // weights for matmuls. note dim == n_heads * head_size
    wq: []f32, // n_layers * dim * (n_heads * head_size)
    wk: []f32, // n_layers * dim * (n_kv_heads * head_size)
    wv: []f32, // n_layers * dim * (n_kv_heads * head_size)
    wo: []f32, // n_layers * (n_heads * head_size) * dim
    rms_ffn_weight: []f32, // n_layers * dim
    w1: []f32, // n_layers * dim * hidden_dim
    w2: []f32, // n_layers * hidden_dim * dim
    w3: []f32, // n_layers * dim * hidden_dim
    rms_final_weight: []f32, // dim
    freq_cis_real: []f32, // seq_len * head_size / 2
    freq_cis_imag: []f32, // seq_len * head_size / 2
    wcls: []f32, // vocab_size * dim
};

pub fn readFile(
    optional_allocator: ?std.mem.Allocator,
    path: []const u8,
    config: *Config,
    weights: *Weights,
) !void {
    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const stat = try file.stat();

    var data: []u8 = undefined;

    if (optional_allocator) |allocator| {
        // https://github.com/cgbur/llama2.zig/commit/23c0711308cfe52971834b3c5716e497e7d6dc3d
        data = try allocator.alignedAlloc(u8, std.mem.page_size, stat.size);

        _ = try file.readAll(data);
    } else {
        data = try std.os.mmap(
            null,
            stat.size,
            std.os.PROT.READ,
            std.os.MAP.PRIVATE,
            file.handle,
            0,
        );

        errdefer std.os.munmap(data);
    }

    var config_data: [*]i32 = @alignCast(@ptrCast(data[0..28]));

    const vocab_size: i32 = config_data[5];

    config.* = Config{
        .dim = @intCast(config_data[0]),
        .hidden_dim = @intCast(config_data[1]),
        .n_layers = @intCast(config_data[2]),
        .n_heads = @intCast(config_data[3]),
        .n_kv_heads = @intCast(config_data[4]),
        .vocab_size = std.math.absCast(vocab_size),
        .seq_len = @intCast(config_data[6]), // max_seq_len
    };

    var weights_data: [*]f32 = @alignCast(@ptrCast(data[28..]));

    // https://github.com/karpathy/llama2.c/commit/c3e0d73bd294e1f5e4d17425fac09aaec536400d
    const shared_weights = vocab_size > 0;
    const head_size = config.dim / config.n_heads;
    const token_embedding_table = readFloatSlice(&weights_data, config.vocab_size * config.dim);

    weights.* = Weights{
        .token_embedding_table = token_embedding_table,
        .rms_att_weight = readFloatSlice(&weights_data, config.n_layers * config.dim),
        .wq = readFloatSlice(&weights_data, config.n_layers * config.dim * (config.n_heads * head_size)),
        .wk = readFloatSlice(&weights_data, config.n_layers * config.dim * (config.n_kv_heads * head_size)),
        .wv = readFloatSlice(&weights_data, config.n_layers * config.dim * (config.n_kv_heads * head_size)),
        .wo = readFloatSlice(&weights_data, config.n_layers * (config.n_heads * head_size) * config.dim),
        .rms_ffn_weight = readFloatSlice(&weights_data, config.n_layers * config.dim),
        .w1 = readFloatSlice(&weights_data, config.n_layers * config.dim * config.hidden_dim),
        .w2 = readFloatSlice(&weights_data, config.n_layers * config.hidden_dim * config.dim),
        .w3 = readFloatSlice(&weights_data, config.n_layers * config.dim * config.hidden_dim),
        .rms_final_weight = readFloatSlice(&weights_data, config.dim),
        .freq_cis_real = readFloatSlice(&weights_data, config.seq_len * head_size / 2),
        .freq_cis_imag = readFloatSlice(&weights_data, config.seq_len * head_size / 2),

        .wcls = if (shared_weights)
            token_embedding_table
        else
            readFloatSlice(&weights_data, config.vocab_size * config.dim),
    };
}

fn readFloatSlice(data: *[*]f32, len: usize) []f32 {
    const slice = data.*[0..len];

    data.* += len;

    return slice;
}

test "read TinyStories 260K checkpoint file" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();

    var config: Config = undefined;
    var weights: Weights = undefined;

    try readFile(allocator, "stories260K.bin", &config, &weights);

    try std.testing.expectEqualDeep(Config{
        .dim = 64,
        .hidden_dim = 172,
        .n_layers = 5,
        .n_heads = 8,
        .n_kv_heads = 4,
        .vocab_size = 512,
        .seq_len = 512,
    }, config);
}
