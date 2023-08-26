const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Checkpoint = @import("checkpoint.zig");
const Cli = @import("cli.zig");
const Sampler = @import("sampler.zig");
const Tokenizer = @import("tokenizer.zig");
const Transformer = @import("transformer.zig");

allocator: std.mem.Allocator,
transformer: Transformer,
tokenizer: Tokenizer,
sampler: Sampler,
prompt_tokens: []usize,
n_steps: usize,
timer: bool,

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const transformer = try Transformer.init(allocator, cli);

    errdefer transformer.deinit();

    const vocab_size = transformer.checkpoint.vocab_size;
    const tokenizer = try Tokenizer.init(allocator, cli.tokenizer_path, vocab_size);

    errdefer tokenizer.deinit();

    const sampler = try Sampler.init(allocator, cli, vocab_size);

    errdefer sampler.deinit();

    return Self{
        .allocator = allocator,
        .transformer = transformer,
        .tokenizer = tokenizer,
        .sampler = sampler,
        .prompt_tokens = try tokenizer.encode(allocator, cli.prompt, true, false),
        .n_steps = cli.n_steps,
        .timer = cli.timer,
    };
}

pub fn deinit(self: *const Self) void {
    self.transformer.deinit();
    self.tokenizer.deinit();
    self.sampler.deinit();
    self.allocator.free(self.prompt_tokens);
}

pub fn generate(self: *Self, writer: anytype) !void {
    std.debug.assert(self.prompt_tokens.len > 0);

    var prompt_tokens_offset: usize = 0;
    var current_token: usize = self.prompt_tokens[prompt_tokens_offset];

    prompt_tokens_offset += 1;

    var start_time: i64 = 0;
    var total_time: i64 = 0;
    var next_token: usize = 1;
    var n_steps: usize = 0;

    for (0..self.n_steps) |pos| {
        if (pos > 0) {
            start_time = std.time.milliTimestamp();
        }

        try self.transformer.forward(current_token, pos);

        if (start_time > 0) {
            total_time += std.time.milliTimestamp() - start_time;
        }

        if (prompt_tokens_offset < self.prompt_tokens.len) {
            next_token = self.prompt_tokens[prompt_tokens_offset];
            prompt_tokens_offset += 1;
        } else {
            next_token = self.sampler.sample(self.transformer.logits);
        }

        n_steps += 1;

        if (next_token == 1) {
            break; // the BOS (=1) token delimits sequences
        }

        const word = self.tokenizer.decode(current_token, next_token);

        try lib.print(word, writer);

        current_token = next_token;
    }

    if (total_time > 0 and self.timer) {
        const average_time = @as(f32, @floatFromInt(total_time)) / @as(f32, @floatFromInt(n_steps));

        try writer.print("\n\nachieved: {d:.3} tok/s\n", .{@as(f32, 1000 / average_time)});
    } else {
        try writer.print("\n", .{});
    }
}

test "generate tiny story" {
    var output = std.ArrayList(u8).init(std.testing.allocator);

    defer output.deinit();

    var arg_iterator = try std.process.argsWithAllocator(std.testing.allocator);

    defer arg_iterator.deinit();

    const cli = Cli{
        .checkpoint_path = "stories260K.bin",
        .temperature = 1,
        .top_p = 0.9,
        .random_seed = 42,
        .n_steps = 10,
        .prompt = "There was",
        .tokenizer_path = "tok512.bin",
        .mmap = false,
        .timer = false,
        .arg_iterator = arg_iterator,
    };

    var generator = try Self.init(std.testing.allocator, &cli);

    defer generator.deinit();

    try generator.generate(output.writer());

    try std.testing.expectEqualStrings("There was a good room\n", output.items);
}
