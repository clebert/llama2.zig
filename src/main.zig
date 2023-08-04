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
    const stdout = std.io.getStdOut().writer();

    var config: checkpoint.Config = undefined;
    var weights: checkpoint.Weights = undefined;

    try checkpoint.readFile(allocator, "stories110M.bin", &config, &weights);
    // try stdout.print("{}\n", .{config});

    var vocab: [][]u8 = try allocator.alloc([]u8, config.vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, config.vocab_size);

    const max_word_length = try tokenizer.readFile(allocator, "tokenizer.bin", vocab, word_scores);
    _ = max_word_length; // TODO

    var run_state: transformer.RunState = undefined;

    try transformer.allocRunState(allocator, config, &run_state);

    var token: usize = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    var next: usize = 1; // TODO

    try stdout.print("<s>\n", .{}); // explicit print the initial BOS token for stylistic symmetry reasons

    const start = std.time.milliTimestamp();
    const steps = 256; // config.seq_len;

    for (0..steps) |pos| {
        // forward the transformer to get logits for the next token
        transformer.run(token, pos, config, &run_state, &weights);

        next = argmax(run_state.logits);

        // following BOS token (1), sentencepiece decoder strips any leading whitespace
        const word = if (token == 1 and vocab[next][0] == ' ') vocab[next][1..] else vocab[next];

        try stdout.print("{s}", .{word});

        token = next;
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
