const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Cli = @import("cli.zig");
const Sampler = @import("sampler.zig");
const Tokenizer = @import("tokenizer.zig");
const Transformer = @import("transformer.zig");

allocator: std.mem.Allocator,
transformer: Transformer,
tokenizer: Tokenizer,
sampler: Sampler,
prompt_tokens: []usize,
timer: bool,

pub fn init(allocator: std.mem.Allocator, cli: *const Cli) !Self {
    const transformer = try Transformer.init(allocator, cli);

    errdefer transformer.deinit();

    const vocab_size = transformer.checkpoint.vocab_size;
    const tokenizer = try Tokenizer.init(allocator, cli.tokenizer_path, vocab_size);

    errdefer tokenizer.deinit();

    const sampler = try Sampler.init(allocator, cli, vocab_size);

    errdefer sampler.deinit();

    const prompt_tokens = try tokenizer.encode(allocator, cli.prompt);

    return Self{
        .allocator = allocator,
        .transformer = transformer,
        .tokenizer = tokenizer,
        .sampler = sampler,
        .prompt_tokens = prompt_tokens,
        .timer = cli.timer,
    };
}

pub fn deinit(self: *const Self) void {
    self.transformer.deinit();
    self.tokenizer.deinit();
    self.sampler.deinit();
    self.allocator.free(self.prompt_tokens);
}

const bos_token = 1; // beginning of sequence
const eos_token = 2; // end of sequence

pub fn generate(self: *Self, writer: anytype) !void {
    var token: usize = bos_token;
    var next_token: usize = 0;
    var prompt_tokens_index: usize = 0;
    var n_timed_steps: usize = 0;
    var start_time: i64 = 0;
    var total_time: i64 = 0;

    for (0..self.transformer.sequence_length) |position| {
        if (position > 0) {
            n_timed_steps += 1;
            start_time = std.time.milliTimestamp();
        }

        try self.transformer.forward(token, position);

        if (start_time > 0) {
            total_time += std.time.milliTimestamp() - start_time;
        }

        if (prompt_tokens_index < self.prompt_tokens.len) {
            next_token = self.prompt_tokens[prompt_tokens_index];
            prompt_tokens_index += 1;
        } else {
            next_token = self.sampler.sample(self.transformer.logits);
        }

        if (next_token == bos_token or next_token == eos_token) {
            break;
        }

        const word = self.tokenizer.decode(next_token, token == bos_token);

        try lib.print(word, writer);

        token = next_token;
    }

    if (n_timed_steps > 0 and self.timer) {
        const average_time =
            @as(f32, @floatFromInt(total_time)) / @as(f32, @floatFromInt(n_timed_steps));

        try writer.print("\n\nachieved: {d:.3} tok/s", .{@as(f32, 1000 / average_time)});
    }

    try writer.print("\n", .{});
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
        .chat = false,
        .system_prompt = "",
        .timer = false,
        .arg_iterator = arg_iterator,
    };

    var generator = try Self.init(std.testing.allocator, &cli);

    defer generator.deinit();

    try generator.generate(output.writer());

    try std.testing.expectEqualStrings("There was a good room\n", output.items);
}
