const std = @import("std");
const Checkpoint = @import("checkpoint.zig");
const Cli = @import("cli.zig");
const lib = @import("lib.zig");
const Tokenizer = @import("tokenizer.zig");
const Transformer = @import("transformer.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const stdout = std.io.getStdOut().writer();

    var cli: Cli = undefined;

    try cli.init(allocator);
    defer cli.deinit();

    var checkpoint: Checkpoint = undefined;

    if (cli.mmap) {
        try checkpoint.initMmapFile(cli.checkpoint_path);
    } else {
        try checkpoint.initReadFile(allocator, cli.checkpoint_path);
    }

    defer checkpoint.deinit();

    if (cli.n_steps == 0) {
        cli.n_steps = checkpoint.seq_len;
    }

    const vocab_size = checkpoint.vocab_size;

    var tokenizer: Tokenizer = undefined;

    try tokenizer.init(allocator, cli.tokenizer_path, vocab_size);
    defer tokenizer.deinit();

    var prompt_tokens = try tokenizer.encode(allocator, cli.input_prompt, true, false);

    defer allocator.free(prompt_tokens);

    var transformer: Transformer = undefined;

    try transformer.init(allocator, &checkpoint);
    defer transformer.deinit();

    var current_token: usize = prompt_tokens[0];

    prompt_tokens = prompt_tokens[1..];

    var next_token: usize = 0; // TODO: null
    var rng_state = cli.random_seed;

    var probability_index_pairs_buffer: []lib.ProbabilityIndexPair =
        try allocator.alloc(lib.ProbabilityIndexPair, vocab_size);

    var step: usize = 0;

    var start_time: i64 = 0;
    var total_time: i64 = 0;

    for (0..@min(cli.n_steps, checkpoint.seq_len)) |pos| {
        if (pos > 0) {
            start_time = std.time.milliTimestamp();
        }

        try transformer.forward(current_token, pos);

        if (start_time > 0) {
            total_time += std.time.milliTimestamp() - start_time;
        }

        if (prompt_tokens.len > 0) {
            next_token = prompt_tokens[0];
            prompt_tokens = prompt_tokens[1..];
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

        step += 1;

        // https://github.com/karpathy/llama2.c/blob/c7a26264a233c32f396b1c67be4ac019d2d8a659/run.c#L765
        if (next_token == 1) {
            break;
        }

        const word = tokenizer.decode(current_token, next_token);

        // https://github.com/karpathy/llama2.c/blob/c7a26264a233c32f396b1c67be4ac019d2d8a659/run.c#L427
        if (word.len == 6 and std.mem.eql(u8, word[0..3], "<0x") and word[5] == '>') {
            const byte: ?u8 = std.fmt.parseInt(u8, word[3..5], 16) catch null;

            if (byte) |char| {
                if (std.ascii.isPrint(char) or std.ascii.isWhitespace(char)) {
                    try stdout.print("{s}", .{[_]u8{char}});
                }
            } else {
                try stdout.print("{s}", .{word});
            }
        } else {
            try stdout.print("{s}", .{word});
        }

        current_token = next_token;
    }

    if (total_time > 0 and !cli.test_mode) {
        const average_time = @as(f32, @floatFromInt(total_time)) / @as(f32, @floatFromInt(step));

        try stdout.print("\n\nachieved: {d:.3} tok/s\n", .{@as(f32, 1000 / average_time)});
    } else {
        try stdout.print("\n", .{});
    }
}

test {
    std.testing.refAllDecls(@This());
}
