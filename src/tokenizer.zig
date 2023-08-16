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
    const sorted_vocab = try sortVocab(allocator, vocab);

    var tokens = try encodeCodepoints(allocator, text, sorted_vocab);
    var double_word_buffer = try allocator.alloc(u8, max_word_length * 2);

    while (mergeBestWordPair(vocab, sorted_vocab, word_scores, tokens, double_word_buffer)) {
        tokens = tokens[0 .. tokens.len - 1];
    }

    return tokens;
}

const VocabEntry = struct { word: []const u8, token: usize };

fn sortVocab(allocator: std.mem.Allocator, vocab: []const []const u8) ![]VocabEntry {
    var array = std.ArrayList(VocabEntry).init(allocator);

    for (vocab, 0..) |word, token| {
        try array.append(VocabEntry{ .word = word, .token = token });
    }

    var slice = try array.toOwnedSlice();

    std.sort.block(VocabEntry, slice, {}, lessThan);

    return slice;
}

fn lessThan(context: void, lhs: VocabEntry, rhs: VocabEntry) bool {
    _ = context;

    return std.mem.lessThan(u8, lhs.word, rhs.word);
}

fn encodeCodepoints(
    allocator: std.mem.Allocator,
    text: []const u8,
    sorted_vocab: []const VocabEntry,
) ![]usize {
    var tokens = std.ArrayList(usize).init(allocator);
    var text_view = try std.unicode.Utf8View.init(text);
    var text_iterator = text_view.iterator();
    var token_index: usize = 0;

    while (text_iterator.nextCodepointSlice()) |codepoints| : (token_index += 1) {
        if (token_index == 0) {
            try tokens.append(lookupToken(" ", sorted_vocab) orelse return error.UnknownCharacter);
        }

        if (lookupToken(codepoints, sorted_vocab)) |token| {
            try tokens.append(token);
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3

            for (codepoints) |codepoint| {
                try tokens.append(@as(usize, codepoint) + 3);
            }
        }
    }

    return tokens.toOwnedSlice();
}

fn lookupToken(word: []const u8, sorted_vocab: []const VocabEntry) ?usize {
    var left: usize = 0;
    var right = sorted_vocab.len;

    while (left < right) {
        const mid = left + (right - left) / 2;
        const vocab_entry = sorted_vocab[mid];

        if (std.mem.eql(u8, vocab_entry.word, word)) {
            return vocab_entry.token;
        }

        if (std.mem.lessThan(u8, vocab_entry.word, word)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return null;
}

fn mergeBestWordPair(
    vocab: []const []const u8,
    sorted_vocab: []const VocabEntry,
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
        const word1 = vocab[tokens[token_index]];
        const word2 = vocab[tokens[token_index + 1]];

        @memcpy(double_word_buffer[0..word1.len], word1);
        @memcpy(double_word_buffer[word1.len..(word1.len + word2.len)], word2);

        const token = lookupToken(
            double_word_buffer[0..(word1.len + word2.len)],
            sorted_vocab,
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

// https://github.com/karpathy/llama2.c/pull/226
// https://github.com/karpathy/llama2.c/pull/297
test "utf-8 support" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const text = "Lets try ö & 株式会社";
    const vocab_size = 32000;

    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, vocab_size);

    const max_word_length = try readFile(allocator, "tokenizer.bin", vocab, word_scores);
    const expected = [_]usize{ 365, 1691, 1018, 3963, 669, 29871, 31409, 30607, 30437, 30564 };
    const actual = try encodeWords(allocator, text, vocab, word_scores, max_word_length);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "empty string" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const text = "";
    const vocab_size = 32000;

    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, vocab_size);

    const max_word_length = try readFile(allocator, "tokenizer.bin", vocab, word_scores);
    const expected = [_]usize{};
    const actual = try encodeWords(allocator, text, vocab, word_scores, max_word_length);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "byte fallback" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const text = "𒎗𓐍";
    const vocab_size = 32000;

    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, vocab_size);

    const max_word_length = try readFile(allocator, "tokenizer.bin", vocab, word_scores);
    const expected = [_]usize{ 29871, 243, 149, 145, 154, 243, 150, 147, 144 };
    const actual = try encodeWords(allocator, text, vocab, word_scores, max_word_length);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "one char tokens" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    defer arena.deinit();

    const allocator = arena.allocator();
    const text = "abcdefgh";
    const vocab_size = 512;

    var vocab: [][]u8 = try allocator.alloc([]u8, vocab_size);
    var word_scores: []f32 = try allocator.alloc(f32, vocab_size);

    const max_word_length = try readFile(allocator, "tok512.bin", vocab, word_scores);
    const expected = [_]usize{ 261, 430, 429, 418, 411, 431, 428, 415 };
    const actual = try encodeWords(allocator, text, vocab, word_scores, max_word_length);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}
