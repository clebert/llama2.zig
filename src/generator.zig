const Self = @This();

const std = @import("std");
const GeneratorArgs = @import("generator_args.zig");
const print = @import("print.zig").print;
const Sampler = @import("sampler.zig");
const Tokenizer = @import("tokenizer.zig");
const Transformer = @import("transformer.zig");

transformer: Transformer,
tokenizer: Tokenizer,
sampler: Sampler,
prompt_tokens: []usize,
verbose: bool,

pub fn createLeaky(allocator: std.mem.Allocator, args: GeneratorArgs) !Self {
    const transformer = try Transformer.createLeaky(allocator, args);
    const vocab_size = transformer.checkpoint.vocab_size;
    const tokenizer = try Tokenizer.readLeaky(allocator, args.model_path, vocab_size);

    return .{
        .transformer = transformer,
        .tokenizer = tokenizer,
        .sampler = try Sampler.createLeaky(allocator, args, vocab_size),
        .prompt_tokens = try tokenizer.encode(allocator, args.prompt),
        .verbose = args.verbose,
    };
}

const bos_token = 1; // beginning of sequence
const eos_token = 2; // end of sequence

pub fn generate(self: *Self, writer: anytype) !void {
    var token: usize = bos_token;
    var next_token: usize = 0;
    var prompt_tokens_index: usize = 0;
    var n_timed_positions: usize = 0;
    var start_time: i64 = 0;
    var total_time: i64 = 0;

    for (0..self.transformer.sequence_length) |position| {
        if (position > 0) {
            n_timed_positions += 1;
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
            next_token = self.sampler.sample(self.transformer.output.data);
        }

        if (next_token == bos_token or next_token == eos_token) {
            break;
        }

        const word = self.tokenizer.decode(next_token, token == bos_token);

        try print(word, writer);

        token = next_token;
    }

    if (n_timed_positions > 0 and self.verbose) {
        const average_time =
            @as(f32, @floatFromInt(total_time)) / @as(f32, @floatFromInt(n_timed_positions));

        try writer.print("\n\nachieved: {d:.3} tok/s", .{@as(f32, 1000 / average_time)});
    }

    try writer.print("\n", .{});
}

test "generate tiny story" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);

    defer arena.deinit();

    var output = std.ArrayList(u8).init(arena.allocator());

    const args = GeneratorArgs{
        .model_path = "models/tinystories_260k",
        .prompt = "There was",
        .random_seed = 42,
        .sequence_length = 10,
        .temperature = 1,
        .thread_count = 0,
        .top_p = 0.9,
        .verbose = false,
    };

    var generator = try Self.createLeaky(arena.allocator(), args);

    try generator.generate(output.writer());

    try std.testing.expectEqualStrings("There was a good room\n", output.items);
}
