const std = @import("std");

const reader = @import("reader.zig");

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
    wq: []f32, // n_layers * dim * dim
    wk: []f32, // n_layers * dim * dim
    wv: []f32, // n_layers * dim * dim
    wo: []f32, // n_layers * dim * dim
    rms_ffn_weight: []f32, // n_layers * dim
    w1: []f32, // n_layers * dim * hidden_dim
    w2: []f32, // n_layers * hidden_dim * dim
    w3: []f32, // n_layers * dim * hidden_dim
    rms_final_weight: []f32, // dim
    freq_cis_real: []f32, // seq_len * (dim / n_heads) / 2
    freq_cis_imag: []f32, // seq_len * (dim / n_heads) / 2
    wcls: []f32, // vocab_size * dim
};

pub fn readFile(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: *Config,
    weights: *Weights,
) !void {
    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);

    std.debug.print("read file {s}\n", .{path});

    _ = try file.readAll(data);

    var offset: usize = 0;

    const dim = reader.readInt(u32, &offset, data);
    const hidden_dim = reader.readInt(u32, &offset, data);
    const n_layers = reader.readInt(u32, &offset, data);
    const n_heads = reader.readInt(u32, &offset, data);
    const n_kv_heads = reader.readInt(u32, &offset, data);
    const vocab_size = reader.readInt(i32, &offset, data);
    const seq_len = reader.readInt(u32, &offset, data);

    config.* = Config{
        .dim = dim,
        .hidden_dim = hidden_dim,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .vocab_size = std.math.absCast(vocab_size),
        .seq_len = seq_len,
    };

    // https://github.com/karpathy/llama2.c/commit/c3e0d73bd294e1f5e4d17425fac09aaec536400d
    const shared_weights = vocab_size > 0;
    const head_size = config.dim / config.n_heads;

    var weights_data: [*]f32 = @alignCast(@ptrCast(data[offset..]));

    var slice_len = config.vocab_size * config.dim;
    const token_embedding_table = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.dim;
    const rms_att_weight = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.dim * config.dim;
    const wq = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.dim * config.dim;
    const wk = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.dim * config.dim;
    const wv = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.dim * config.dim;
    const wo = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.dim;
    const rms_ffn_weight = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.dim * config.hidden_dim;
    const w1 = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.hidden_dim * config.dim;
    const w2 = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.n_layers * config.dim * config.hidden_dim;
    const w3 = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.dim;
    const rms_final_weight = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.seq_len * head_size / 2;
    const freq_cis_real = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.seq_len * head_size / 2;
    const freq_cis_imag = weights_data[0..slice_len];
    weights_data += slice_len;

    slice_len = config.vocab_size * config.dim;
    const wcls = if (shared_weights) token_embedding_table else weights_data[0..slice_len];

    weights.* = Weights{
        .token_embedding_table = token_embedding_table,
        .rms_att_weight = rms_att_weight,
        .wq = wq,
        .wk = wk,
        .wv = wv,
        .wo = wo,
        .rms_ffn_weight = rms_ffn_weight,
        .w1 = w1,
        .w2 = w2,
        .w3 = w3,
        .rms_final_weight = rms_final_weight,
        .freq_cis_real = freq_cis_real,
        .freq_cis_imag = freq_cis_imag,
        .wcls = wcls,
    };
}

test "read TinyStories 15M checkpoint file" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();

    var config: Config = undefined;
    var weights: Weights = undefined;

    readFile(allocator, "stories15M.bin", &config, &weights) catch |err| {
        if (err == error.FileNotFound) return else return err;
    };

    try std.testing.expectEqualDeep(config, Config{
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 32000,
        .seq_len = 256,
    });
}
