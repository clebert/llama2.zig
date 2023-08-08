const std = @import("std");

const checkpoint = @import("checkpoint.zig");
const cli = @import("cli.zig");
const tokenizer = @import("tokenizer.zig");
const transformer = @import("transformer.zig");
const utils = @import("utils.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const stdout = std.io.getStdOut().writer();

    var args = try cli.parseArgs(allocator);
    var config: checkpoint.Config = undefined;
    var weights: checkpoint.Weights = undefined;

    try checkpoint.readFile(allocator, args.checkpoint_path, &config, &weights);
    try stdout.print("{}\n", .{config});

    if (args.n_steps == 0) {
        args.n_steps = config.seq_len;
    }

    var vocab: [][]u8 = try allocator.alloc([]u8, config.vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, config.vocab_size);

    const max_word_length = try tokenizer.readFile(allocator, "tokenizer.bin", vocab, word_scores);

    var prompt_tokens = try tokenizer.encodeWords(
        allocator,
        args.input_prompt,
        vocab,
        word_scores,
        max_word_length,
    );

    var run_state: transformer.RunState = undefined;

    try transformer.allocRunState(allocator, config, &run_state);

    var token: usize = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    var next: usize = 1; // TODO

    try stdout.print("<s>\n", .{}); // explicit print the initial BOS token for stylistic symmetry reasons

    var start: i64 = 0;
    var rng = std.rand.DefaultPrng.init(args.random_seed);
    var prob_indices: []utils.ProbIndex = try allocator.alloc(utils.ProbIndex, config.vocab_size);

    for (0..args.n_steps) |pos| {
        // forward the transformer to get logits for the next token
        try transformer.run(token, pos, config, &run_state, &weights);

        if (prompt_tokens.len > 0) {
            next = prompt_tokens[0];

            prompt_tokens = prompt_tokens[1..];
        } else if (args.temperature == 0) {
            next = utils.argmax(run_state.logits);
        } else {
            // apply the temperature to the logits
            for (run_state.logits) |*logit| {
                logit.* /= args.temperature;
            }

            // apply softmax to the logits to get the probabilities for next token
            utils.softmax(run_state.logits);

            if (args.top_p == 0) {
                // we sample from this distribution to get the next token
                next = utils.sample(&rng, run_state.logits);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = utils.sampleTopP(&rng, run_state.logits, args.top_p, prob_indices);
            }
        }

        // following BOS token (1), sentencepiece decoder strips any leading whitespace
        const word = if (token == 1 and vocab[next][0] == ' ') vocab[next][1..] else vocab[next];

        try stdout.print("{s}", .{word});

        token = next;

        if (start == 0) {
            start = std.time.milliTimestamp();
        }
    }

    // report achieved tok/s
    const end = std.time.milliTimestamp();
    const step_cast: i64 = @intCast(args.n_steps - 1);
    const tokps: i64 = @divFloor(step_cast * 1000, end - start);

    try stdout.print("\nachieved tok/s: {}\n", .{tokps});
}

test {
    std.testing.refAllDecls(@This());
}
