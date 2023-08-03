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
    wcls: []f32, // TODO: vocab_size * dim
};

pub fn readFile(allocator: std.mem.Allocator, path: []const u8, config: *Config, weights: *Weights) !void {
    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);

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

    // https://github.com/karpathy/llama2.c/commit/c3e0d73bd294e1f5e4d17425fac09aaec536400d#diff-8935a7a088435e2ddf7315451f07fae16810932fb3a0a5d706a2eead1618af26R402
    const shared_weights = vocab_size > 0;
    const token_embedding_table = try reader.readFloatSlice(allocator, config.vocab_size * config.dim, &offset, data);
    const head_size = config.dim / config.n_heads;

    weights.* = Weights{
        .token_embedding_table = token_embedding_table,
        .rms_att_weight = try reader.readFloatSlice(allocator, config.n_layers * config.dim, &offset, data),
        .wq = try reader.readFloatSlice(allocator, config.n_layers * config.dim * config.dim, &offset, data),
        .wk = try reader.readFloatSlice(allocator, config.n_layers * config.dim * config.dim, &offset, data),
        .wv = try reader.readFloatSlice(allocator, config.n_layers * config.dim * config.dim, &offset, data),
        .wo = try reader.readFloatSlice(allocator, config.n_layers * config.dim * config.dim, &offset, data),
        .rms_ffn_weight = try reader.readFloatSlice(allocator, config.n_layers * config.dim, &offset, data),
        .w1 = try reader.readFloatSlice(allocator, config.n_layers * config.dim * config.hidden_dim, &offset, data),
        .w2 = try reader.readFloatSlice(allocator, config.n_layers * config.hidden_dim * config.dim, &offset, data),
        .w3 = try reader.readFloatSlice(allocator, config.n_layers * config.dim * config.hidden_dim, &offset, data),
        .rms_final_weight = try reader.readFloatSlice(allocator, config.dim, &offset, data),
        .freq_cis_real = try reader.readFloatSlice(allocator, config.seq_len * head_size / 2, &offset, data),
        .freq_cis_imag = try reader.readFloatSlice(allocator, config.seq_len * head_size / 2, &offset, data),
        .wcls = if (shared_weights) token_embedding_table else try reader.readFloatSlice(allocator, config.vocab_size * config.dim, &offset, data),
    };
}
