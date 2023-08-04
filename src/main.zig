const std = @import("std");

const checkpoint = @import("checkpoint.zig");
const tokenizer = @import("tokenizer.zig");
const transformer = @import("transformer.zig");

// https://github.com/karpathy/llama2.c/blob/af8708d87bcda7fda97b93f6c135dd43ea78106c/run.c
// without omp support
pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();

    // zig build run -Doptimize=ReleaseFast -- <checkpoint_file_path> <steps>=seq_len <prompt>=""
    var args = try std.process.argsWithAllocator(allocator);
    var checkpoint_file_path: ?[]const u8 = null;
    var steps: usize = 0;
    var prompt: []const u8 = "";
    var arg_index: usize = 0;

    while (args.next()) |arg| {
        if (arg_index == 1) {
            checkpoint_file_path = arg;
        } else if (arg_index == 2) {
            steps = try std.fmt.parseInt(usize, arg, 10);
        } else if (arg_index == 3) {
            prompt = arg;
        }

        arg_index += 1;
    }

    const stdout = std.io.getStdOut().writer();

    var config: checkpoint.Config = undefined;
    var weights: checkpoint.Weights = undefined;

    try checkpoint.readFile(allocator, checkpoint_file_path orelse return error.MissingCheckpointFile, &config, &weights);
    try stdout.print("{}\n", .{config});

    if (steps == 0) {
        steps = config.seq_len;
    }

    var vocab: [][]u8 = try allocator.alloc([]u8, config.vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, config.vocab_size);

    const max_word_length = try tokenizer.readFile(allocator, "tokenizer.bin", vocab, word_scores);
    var prompt_tokens = try tokenizer.encodeWords(allocator, prompt, vocab, word_scores, max_word_length);

    var run_state: transformer.RunState = undefined;

    try transformer.allocRunState(allocator, config, &run_state);

    var token: usize = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    var next: usize = 1; // TODO

    try stdout.print("<s>\n", .{}); // explicit print the initial BOS token for stylistic symmetry reasons

    var start: i64 = 0;

    for (0..steps) |pos| {
        // forward the transformer to get logits for the next token
        transformer.run(token, pos, config, &run_state, &weights);

        if (prompt_tokens.len > 0) {
            next = prompt_tokens[0];

            prompt_tokens = prompt_tokens[1..];
        } else {
            next = argmax(run_state.logits);
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
    const step_cast: i64 = @intCast(steps - 1);
    const tokps: i64 = @divFloor(step_cast * 1000, end - start);

    try stdout.print("\nachieved tok/s: {}\n", .{tokps});
}

fn argmax(v: []f32) usize {
    // return argmax of v in elements 0..n
    var max_i: usize = 0;
    var max_p: f32 = v[0];

    for (1..v.len) |i| {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }

    return max_i;
}

test {
    std.testing.refAllDecls(@This());
}
