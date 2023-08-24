const std = @import("std");

const Self = @This();

allocator: ?std.mem.Allocator,
dim: usize,
hidden_dim: usize,
n_layers: usize,
n_heads: usize,
n_kv_heads: usize,
vocab_size: usize,
seq_len: usize,
kv_dim: usize,
head_size: usize,
head_size_sqrt: f32,
n_groups: usize,

weights: struct {
    token_embedding: []const f32,

    attention_input_rms: []const f32,
    attention_query: []const f32,
    attention_key: []const f32,
    attention_value: []const f32,
    attention_output: []const f32,

    ffn_input_rms: []const f32,
    ffn_input_to_hidden: []const f32, // w1
    ffn_hidden_to_output: []const f32, // w2
    ffn_input_to_residual: []const f32, // w3

    final_rms: []const f32,
    classifier: []const f32,
},

data: []align(std.mem.page_size) const u8,

pub fn initReadFile(self: *Self, allocator: std.mem.Allocator, path: []const u8) !void {
    self.allocator = allocator;

    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const stat = try file.stat();

    var data = try allocator.alignedAlloc(u8, std.mem.page_size, stat.size);

    errdefer allocator.free(data);

    const n_bytes_read = try file.readAll(data);

    if (n_bytes_read != data.len) {
        return error.UnexpectedEndOfFile;
    }

    self.data = data;

    self.init();
}

pub fn initMmapFile(self: *Self, path: []const u8) !void {
    self.allocator = null;

    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const stat = try file.stat();

    self.data = try std.os.mmap(
        null,
        stat.size,
        std.os.PROT.READ,
        std.os.MAP.PRIVATE,
        file.handle,
        0,
    );

    errdefer std.os.munmap(self.data);

    self.init();
}

fn init(self: *Self) void {
    var config_data: [*]const i32 = @alignCast(@ptrCast(self.data[0..28]));

    const signed_vocab_size: i32 = config_data[5];

    self.dim = @intCast(config_data[0]);
    self.hidden_dim = @intCast(config_data[1]);
    self.n_layers = @intCast(config_data[2]);
    self.n_heads = @intCast(config_data[3]);
    self.n_kv_heads = @intCast(config_data[4]);
    self.vocab_size = std.math.absCast(signed_vocab_size);
    self.seq_len = @intCast(config_data[6]);
    self.kv_dim = (self.dim * self.n_kv_heads) / self.n_heads;
    self.head_size = self.dim / self.n_heads;
    self.head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(self.head_size)));
    self.n_groups = self.n_heads / self.n_kv_heads;

    const dim = self.dim;
    const n_layers = self.n_layers;
    const n_heads = self.n_heads;
    const n_kv_heads = self.n_kv_heads;

    var weights_data: [*]const f32 = @alignCast(@ptrCast(self.data[28..]));

    const token_embedding = readFloatSlice(&weights_data, self.vocab_size * dim);

    const head_size = dim / n_heads;
    const attention_input_rms = readFloatSlice(&weights_data, n_layers * dim);
    const attention_query = readFloatSlice(&weights_data, n_layers * dim * n_heads * head_size);
    const attention_key = readFloatSlice(&weights_data, n_layers * dim * n_kv_heads * head_size);
    const attention_value = readFloatSlice(&weights_data, n_layers * dim * n_kv_heads * head_size);
    const attention_output = readFloatSlice(&weights_data, n_layers * n_heads * head_size * dim);

    const ffn_input_rms = readFloatSlice(&weights_data, n_layers * dim);
    const ffn_input_to_hidden = readFloatSlice(&weights_data, n_layers * dim * self.hidden_dim);
    const ffn_hidden_to_output = readFloatSlice(&weights_data, n_layers * self.hidden_dim * dim);
    const ffn_input_to_residual = readFloatSlice(&weights_data, n_layers * dim * self.hidden_dim);

    const final_rms = readFloatSlice(&weights_data, dim);

    _ = readFloatSlice(&weights_data, self.seq_len * head_size / 2);
    _ = readFloatSlice(&weights_data, self.seq_len * head_size / 2);

    self.weights = .{
        .token_embedding = token_embedding,

        .attention_input_rms = attention_input_rms,
        .attention_query = attention_query,
        .attention_key = attention_key,
        .attention_value = attention_value,
        .attention_output = attention_output,

        .ffn_input_rms = ffn_input_rms,
        .ffn_input_to_hidden = ffn_input_to_hidden,
        .ffn_hidden_to_output = ffn_hidden_to_output,
        .ffn_input_to_residual = ffn_input_to_residual,

        .final_rms = final_rms,

        // https://github.com/karpathy/llama2.c/commit/c3e0d73bd294e1f5e4d17425fac09aaec536400d
        .classifier = if (signed_vocab_size > 0)
            token_embedding
        else
            readFloatSlice(&weights_data, self.vocab_size * dim),
    };
}

pub fn deinit(self: *const Self) void {
    if (self.allocator) |allocator| {
        allocator.free(self.data);
    } else {
        std.os.munmap(self.data);
    }
}

fn readFloatSlice(data: *[*]const f32, len: usize) []const f32 {
    const slice = data.*[0..len];

    data.* += len;

    return slice;
}
