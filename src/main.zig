const std = @import("std");

const Checkpoint = @import("checkpoint.zig").Checkpoint;
const Cli = @import("cli.zig").Cli;
const lib = @import("lib.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Transformer = @import("transformer.zig").Transformer;

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
    var first_decoding_time: i64 = 0;
    var total_decoding_time: i64 = 0;
    var total_sampling_time: i64 = 0;

    // advance the state state machine
    for (0..@min(cli.n_steps, checkpoint.seq_len)) |pos| {
        start_time = std.time.milliTimestamp();

        try transformer.forward(current_token, pos);

        if (pos == 0) {
            first_decoding_time = std.time.milliTimestamp() - start_time;
            total_decoding_time = first_decoding_time;
        } else {
            total_decoding_time += std.time.milliTimestamp() - start_time;
        }

        start_time = std.time.milliTimestamp();

        if (prompt_tokens.len > 0) {
            next_token = prompt_tokens[0];

            prompt_tokens = prompt_tokens[1..];
        } else if (cli.temperature == 0) {
            next_token = lib.argmax(transformer.logits);
        } else {
            // apply the temperature to the logits
            for (transformer.logits) |*logit| {
                logit.* /= cli.temperature;
            }

            // apply softmax to the logits to get the probabilities for next token
            lib.softmax(transformer.logits);

            if (cli.top_p <= 0 or cli.top_p >= 1) {
                // we sample from this distribution to get the next token
                next_token = lib.sampleMultinomial(lib.random(&rng_state), transformer.logits);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next_token = lib.sampleNucleus(
                    lib.random(&rng_state),
                    transformer.logits,
                    cli.top_p,
                    probability_index_pairs_buffer,
                );
            }
        }

        total_sampling_time += std.time.milliTimestamp() - start_time;
        step += 1;

        // data-dependent terminating condition: the BOS (1) token delimits sequences
        if (next_token == 1) {
            break;
        }

        const word = tokenizer.decode(current_token, next_token);

        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        if (word.len == 6 and std.mem.eql(u8, word[0..3], "<0x") and word[5] == '>') {
            const byte: ?u8 = std.fmt.parseInt(u8, word[3..5], 16) catch null;

            if (byte) |char| {
                // ok this token is a raw byte token, carefuly to only print printable chars or whitespace
                // some of the other bytes can be various control codes, backspace, etc. => skip

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

    if (step > 1 and !cli.test_mode) {
        const average_decoding_time: f32 =
            @as(f32, @floatFromInt(total_decoding_time - first_decoding_time)) /
            @as(f32, @floatFromInt(step - 1));

        const average_sampling_time: f32 =
            @as(f32, @floatFromInt(total_sampling_time)) / @as(f32, @floatFromInt(step));

        const tokens_per_second: f32 = 1000 / (average_decoding_time + average_sampling_time);

        try stdout.print("\n\nachieved: {d:.3} tok/s\n\n", .{tokens_per_second});
        try stdout.print("total decoding time: {} ms\n", .{total_decoding_time});
        try stdout.print("average decoding time: {d:.3} ms\n", .{average_decoding_time});
        try stdout.print("first decoding time: {} ms\n", .{first_decoding_time});
        try stdout.print("total sampling time: {} ms\n", .{total_sampling_time});
        try stdout.print("average sampling time: {d:.3} ms\n", .{average_sampling_time});
    } else {
        try stdout.print("\n", .{});
    }
}

test {
    std.testing.refAllDecls(@This());
}
