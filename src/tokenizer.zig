const std = @import("std");

pub fn readFile(
    allocator: std.mem.Allocator,
    path: []const u8,
    vocab: [][]u8,
    word_scores: []f32,
) !usize {
    const file = try std.fs.cwd().openFile(path, .{});

    defer file.close();

    const reader = file.reader();
    const max_word_length = try reader.readIntLittle(u32);

    for (word_scores, 0..) |*word_score, word_index| {
        word_score.* = @bitCast(try reader.readIntLittle(u32));

        const word_length = try reader.readIntLittle(u32);
        const word = try allocator.alloc(u8, word_length);

        try reader.readNoEof(word);

        vocab[word_index] = word;
    }

    return max_word_length;
}

pub fn encodeWords(
    allocator: std.mem.Allocator,
    text: []const u8,
    vocab: []const []const u8,
    word_scores: []const f32,
    max_word_length: usize,
) ![]usize {
    var tokens = try allocator.alloc(usize, text.len);

    try encodeCharacters(text, vocab, tokens);

    var double_word_buffer = try allocator.alloc(u8, max_word_length * 2);

    while (mergeBestWordPair(vocab, word_scores, tokens, double_word_buffer)) {
        tokens = tokens[0 .. tokens.len - 1];
    }

    return tokens;
}

fn encodeCharacters(text: []const u8, vocab: []const []const u8, tokens: []usize) !void {
    for (text, 0..) |char, token_index| {
        tokens[token_index] = lookupToken(
            ([_]u8{char})[0..],
            vocab,
        ) orelse return error.UnknownCharacter;
    }
}

fn lookupToken(word: []const u8, vocab: []const []const u8) ?usize {
    for (vocab, 0..) |vocab_word, token| {
        if (std.mem.eql(u8, word, vocab_word)) return token;
    }

    return null;
}

fn mergeBestWordPair(
    vocab: []const []const u8,
    word_scores: []const f32,
    tokens: []usize,
    double_word_buffer: []u8,
) bool {
    if (tokens.len < 1) {
        return false;
    }

    var best_token: ?usize = null;
    var best_token_index: ?usize = null;
    var best_word_score = -std.math.floatMax(f32);

    for (0..tokens.len - 1) |token_index| {
        const word_1 = vocab[tokens[token_index]];
        const word_2 = vocab[tokens[token_index + 1]];

        @memcpy(double_word_buffer[0..word_1.len], word_1);
        @memcpy(double_word_buffer[word_1.len..(word_1.len + word_2.len)], word_2);

        const token = lookupToken(
            double_word_buffer[0..(word_1.len + word_2.len)],
            vocab,
        ) orelse continue;

        const word_score = word_scores[token];

        if (word_score > best_word_score) {
            best_token = token;
            best_token_index = token_index;
            best_word_score = word_score;
        }
    }

    if (best_token_index) |token_index| {
        std.mem.copyForwards(
            usize,
            tokens[token_index + 1 .. tokens.len - 1],
            tokens[token_index + 2 ..],
        );

        tokens[token_index] = best_token.?;

        return true;
    } else {
        return false;
    }
}

test "encode multiple words" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const text = "One day, Lily met a Shoggoth";
    const vocab_size = 32000;

    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, vocab_size);

    const max_word_length = try readFile(allocator, "tokenizer.bin", vocab, word_scores);
    const expected = [_]usize{ 6716, 2462, 47, 365, 2354, 1539, 263, 1383, 468, 106, 720 };
    const actual = try encodeWords(allocator, text, vocab, word_scores, max_word_length);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
    try std.testing.expectEqualStrings("One", vocab[actual[0]]);
    try std.testing.expectEqualStrings(" day", vocab[actual[1]]);
    try std.testing.expectEqualStrings(",", vocab[actual[2]]);
    try std.testing.expectEqualStrings(" L", vocab[actual[3]]);
    try std.testing.expectEqualStrings("ily", vocab[actual[4]]);
    try std.testing.expectEqualStrings(" met", vocab[actual[5]]);
    try std.testing.expectEqualStrings(" a", vocab[actual[6]]);
    try std.testing.expectEqualStrings(" Sh", vocab[actual[7]]);
    try std.testing.expectEqualStrings("og", vocab[actual[8]]);
    try std.testing.expectEqualStrings("g", vocab[actual[9]]);
    try std.testing.expectEqualStrings("oth", vocab[actual[10]]);
}

test "encode empty string" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const text = "";
    const vocab_size = 32000;

    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, vocab_size);

    const max_word_length = try readFile(allocator, "tokenizer.bin", vocab, word_scores);
    const actual = try encodeWords(allocator, text, vocab, word_scores, max_word_length);

    try std.testing.expectEqual(@as(usize, 0), actual.len);
}

test "encode single character" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const text = "A";
    const vocab_size = 32000;

    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, vocab_size);

    const max_word_length = try readFile(allocator, "tokenizer.bin", vocab, word_scores);
    const expected = [_]usize{68};
    const actual = try encodeWords(allocator, text, vocab, word_scores, max_word_length);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
    try std.testing.expectEqualStrings("A", vocab[actual[0]]);
}
