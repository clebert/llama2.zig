const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const Cli = @import("cli.zig");
const lib = @import("lib.zig");
const Tokenizer = @import("tokenizer.zig");
const Transformer = @import("transformer.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    var cli = try Cli.init(arena.allocator());

    defer cli.deinit();

    const stdout = std.io.getStdOut().writer();

    try generate(arena.allocator(), &cli, stdout);
}

fn generate(allocator: std.mem.Allocator, cli: *const Cli, writer: anytype) !void {
    const checkpoint = try Checkpoint.init(if (cli.mmap) null else allocator, cli.checkpoint_path);

    defer checkpoint.deinit();

    const tokenizer = try Tokenizer.init(allocator, cli.tokenizer_path, checkpoint.vocab_size);

    defer tokenizer.deinit();

    const prompt_tokens = try tokenizer.encode(allocator, cli.input_prompt, true, false);

    defer allocator.free(prompt_tokens);

    const transformer = try Transformer.init(allocator, &checkpoint, cli.n_steps);

    defer transformer.deinit();

    std.debug.assert(prompt_tokens.len > 0);

    var prompt_tokens_offset: usize = 0;
    var current_token: usize = prompt_tokens[prompt_tokens_offset];

    prompt_tokens_offset += 1;

    var probability_index_pairs_buffer: []lib.ProbabilityIndexPair =
        try allocator.alloc(lib.ProbabilityIndexPair, checkpoint.vocab_size);

    defer allocator.free(probability_index_pairs_buffer);

    var start_time: i64 = 0;
    var total_time: i64 = 0;
    var next_token: usize = 1;
    var rng_state = cli.random_seed;
    var n_steps: usize = 0;

    for (0..cli.n_steps) |pos| {
        if (pos > 0) {
            start_time = std.time.milliTimestamp();
        }

        try transformer.forward(current_token, pos);

        if (start_time > 0) {
            total_time += std.time.milliTimestamp() - start_time;
        }

        if (prompt_tokens_offset < prompt_tokens.len) {
            next_token = prompt_tokens[prompt_tokens_offset];
            prompt_tokens_offset += 1;
        } else if (cli.temperature == 0) {
            next_token = lib.argmax(transformer.logits);
        } else {
            for (transformer.logits) |*logit| {
                logit.* /= cli.temperature;
            }

            lib.softmax(transformer.logits);

            if (cli.top_p <= 0 or cli.top_p >= 1) {
                next_token = lib.sampleMultinomial(lib.random(&rng_state), transformer.logits);
            } else {
                next_token = lib.sampleNucleus(
                    lib.random(&rng_state),
                    transformer.logits,
                    cli.top_p,
                    probability_index_pairs_buffer,
                );
            }
        }

        n_steps += 1;

        // https://github.com/karpathy/llama2.c/blob/c7a26264a233c32f396b1c67be4ac019d2d8a659/run.c#L765
        if (next_token == 1) {
            break;
        }

        const word = tokenizer.decode(current_token, next_token);

        try lib.print(word, writer);

        current_token = next_token;
    }

    if (total_time > 0 and !cli.test_mode) {
        const average_time = @as(f32, @floatFromInt(total_time)) / @as(f32, @floatFromInt(n_steps));

        try writer.print("\n\nachieved: {d:.3} tok/s\n", .{@as(f32, 1000 / average_time)});
    } else {
        try writer.print("\n", .{});
    }
}

test {
    std.testing.refAllDecls(@This());
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
        .input_prompt = "There was",
        .tokenizer_path = "tok512.bin",
        .mmap = false,
        .test_mode = true,
        .arg_iterator = arg_iterator,
    };

    try generate(std.testing.allocator, &cli, output.writer());

    try std.testing.expectEqualStrings("There was a good room\n", output.items);
}
