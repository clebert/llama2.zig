const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const stdout = std.io.getStdOut().writer();
    const vocab_size = 32000;

    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, vocab_size);

    const max_word_length = try tokenizer.loadVocab(allocator, "tokenizer.bin", vocab, word_scores);

    try stdout.print("{}\n", .{max_word_length});
}
