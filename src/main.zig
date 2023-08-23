const std = @import("std");

const checkpoint = @import("checkpoint.zig");
const cli = @import("cli.zig");
const lib = @import("lib.zig");
const tokenizer = @import("tokenizer.zig");
const Transformer = @import("transformer.zig").Transformer;
const utils = @import("utils.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const stdout = std.io.getStdOut().writer();

    var args = try cli.parseArgs(allocator);
    var config: checkpoint.Config = undefined;
    var weights: checkpoint.Weights = undefined;

    try checkpoint.readFile(
        if (args.mmap) null else allocator,
        args.checkpoint_path,
        &config,
        &weights,
    );

    if (!args.test_mode) {
        try stdout.print("{}\n\n", .{config});
    }

    if (args.n_steps == 0) {
        args.n_steps = config.seq_len;
    }

    var vocab: [][]u8 = try allocator.alloc([]u8, config.vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, config.vocab_size);

    const max_word_length = try tokenizer.readFile(
        allocator,
        args.tokenizer_path,
        vocab,
        word_scores,
    );

    var prompt_tokens = try tokenizer.encodeWords(
        allocator,
        args.input_prompt,
        vocab,
        word_scores,
        max_word_length,
    );

    var transformer: Transformer = undefined;

    try transformer.init(allocator, &config);
    defer transformer.deinit(allocator);

    var token: usize = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    var next: usize = 1; // TODO
    var rng_state = args.random_seed;
    var prob_indices: []utils.ProbIndex = try allocator.alloc(utils.ProbIndex, config.vocab_size);
    var n_steps: usize = 0;

    var start_time: i64 = 0;
    var first_decoding_time: i64 = 0;
    var total_decoding_time: i64 = 0;
    var total_sampling_time: i64 = 0;

    // advance the state state machine
    for (0..args.n_steps) |pos| {
        start_time = std.time.milliTimestamp();

        try transformer.forward(token, pos, &config, &weights);

        if (pos == 0) {
            first_decoding_time = std.time.milliTimestamp() - start_time;
            total_decoding_time = first_decoding_time;
        } else {
            total_decoding_time += std.time.milliTimestamp() - start_time;
        }

        start_time = std.time.milliTimestamp();

        if (prompt_tokens.len > 0) {
            next = prompt_tokens[0];

            prompt_tokens = prompt_tokens[1..];
        } else if (args.temperature == 0) {
            next = lib.argmax(transformer.logits);
        } else {
            // apply the temperature to the logits
            for (transformer.logits) |*logit| {
                logit.* /= args.temperature;
            }

            // apply softmax to the logits to get the probabilities for next token
            lib.softmax(transformer.logits);

            if (args.top_p <= 0 or args.top_p >= 1) {
                // we sample from this distribution to get the next token
                next = lib.sample(lib.random(&rng_state), transformer.logits);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = utils.sampleTopP(lib.random(&rng_state), transformer.logits, args.top_p, prob_indices);
            }
        }

        total_sampling_time += std.time.milliTimestamp() - start_time;
        n_steps += 1;

        // data-dependent terminating condition: the BOS (1) token delimits sequences
        if (next == 1) {
            break;
        }

        // following BOS (1) token, sentencepiece decoder strips any leading whitespace
        const word = if (token == 1 and vocab[next][0] == ' ') vocab[next][1..] else vocab[next];

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

        token = next;
    }

    if (n_steps > 1 and !args.test_mode) {
        const average_decoding_time: f32 =
            @as(f32, @floatFromInt(total_decoding_time - first_decoding_time)) /
            @as(f32, @floatFromInt(n_steps - 1));

        const average_sampling_time: f32 =
            @as(f32, @floatFromInt(total_sampling_time)) / @as(f32, @floatFromInt(n_steps));

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
