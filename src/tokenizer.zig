const Self = @This();

const std = @import("std");

allocator: std.mem.Allocator,
max_word_length: usize,
vocab: []const []const u8,
word_scores: []const f32,
sorted_vocab: []const VocabEntry,

pub fn init(allocator: std.mem.Allocator, path: []const u8, vocab_size: usize) !Self {
    var vocab = try allocator.alloc([]u8, vocab_size);

    errdefer for (vocab) |word| {
        allocator.free(word);
    };

    errdefer allocator.free(vocab);

    var word_scores = try allocator.alloc(f32, vocab_size);

    errdefer allocator.free(word_scores);

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

    return Self{
        .allocator = allocator,
        .max_word_length = max_word_length,
        .vocab = vocab,
        .word_scores = word_scores,
        .sorted_vocab = try sortVocab(allocator, vocab),
    };
}

pub fn deinit(self: *const Self) void {
    for (self.vocab) |word| {
        self.allocator.free(word);
    }

    self.allocator.free(self.vocab);
    self.allocator.free(self.word_scores);
    self.allocator.free(self.sorted_vocab);
}

pub fn encode(
    self: *const Self,
    allocator: std.mem.Allocator,
    text: []const u8,
    prepend_bos_token: bool,
    append_eos_token: bool,
) ![]usize {
    var double_word_buffer = try allocator.alloc(u8, self.max_word_length * 2);

    defer allocator.free(double_word_buffer);

    var tokens = try self.encodeCodepoints(
        allocator,
        text,
        prepend_bos_token,
        append_eos_token,
    );

    defer allocator.free(tokens);

    var merged_tokens = tokens[0..];

    while (self.mergeBestWordPair(merged_tokens, double_word_buffer)) {
        merged_tokens = merged_tokens[0 .. merged_tokens.len - 1];
    }

    var merged_tokens_copy: []usize = try allocator.alloc(usize, merged_tokens.len);

    @memcpy(merged_tokens_copy, merged_tokens);

    return merged_tokens_copy;
}

pub fn decode(self: *const Self, current_token: usize, next_token: usize) []const u8 {
    // https://github.com/karpathy/llama2.c/blob/7ac65cb2c2b169050747be92011b7bebdd1b4544/run.c#L425
    const word = if (current_token == 1 and self.vocab[next_token][0] == ' ')
        self.vocab[next_token][1..]
    else
        self.vocab[next_token];

    return word;
}

fn encodeCodepoints(
    self: *const Self,
    allocator: std.mem.Allocator,
    text: []const u8,
    prepend_bos_token: bool,
    append_eos_token: bool,
) ![]usize {
    var tokens = std.ArrayList(usize).init(allocator);

    errdefer tokens.deinit();

    var text_view = try std.unicode.Utf8View.init(text);
    var text_iterator = text_view.iterator();
    var token_index: usize = 0;

    if (prepend_bos_token) {
        try tokens.append(1);
    }

    while (text_iterator.nextCodepointSlice()) |codepoints| : (token_index += 1) {
        if (token_index == 0) {
            // https://github.com/karpathy/llama2.c/blob/7ac65cb2c2b169050747be92011b7bebdd1b4544/run.c#L483
            try tokens.append(self.lookupToken(" ") orelse return error.BadVocab);
        }

        if (self.lookupToken(codepoints)) |token| {
            try tokens.append(token);
        } else {
            // https://github.com/karpathy/llama2.c/blob/7ac65cb2c2b169050747be92011b7bebdd1b4544/run.c#L531
            for (codepoints) |codepoint| {
                try tokens.append(@as(usize, codepoint) + 3);
            }
        }
    }

    if (append_eos_token) {
        try tokens.append(2);
    }

    return tokens.toOwnedSlice();
}

fn mergeBestWordPair(self: *const Self, tokens: []usize, double_word_buffer: []u8) bool {
    if (tokens.len < 1) {
        return false;
    }

    var best_token: ?usize = null;
    var best_token_index: ?usize = null;
    var best_word_score = -std.math.floatMax(f32);

    for (0..tokens.len - 1) |token_index| {
        const word1 = self.vocab[tokens[token_index]];
        const word2 = self.vocab[tokens[token_index + 1]];

        @memcpy(double_word_buffer[0..word1.len], word1);
        @memcpy(double_word_buffer[word1.len..(word1.len + word2.len)], word2);

        const token =
            self.lookupToken(double_word_buffer[0..(word1.len + word2.len)]) orelse continue;

        const word_score = self.word_scores[token];

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
    }

    return false;
}

fn lookupToken(self: *const Self, word: []const u8) ?usize {
    var left: usize = 0;
    var right = self.sorted_vocab.len;

    while (left < right) {
        const mid = left + (right - left) / 2;
        const vocab_entry = self.sorted_vocab[mid];

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

const VocabEntry = struct { word: []const u8, token: usize };

fn sortVocab(allocator: std.mem.Allocator, vocab: []const []const u8) ![]VocabEntry {
    var array = std.ArrayList(VocabEntry).init(allocator);

    errdefer array.deinit();

    for (vocab, 0..) |word, token| {
        try array.append(VocabEntry{ .word = word, .token = token });
    }

    var slice = try array.toOwnedSlice();

    // sort entries in ascending order
    std.sort.block(VocabEntry, slice, {}, lessThan);

    return slice;
}

fn lessThan(context: void, lhs: VocabEntry, rhs: VocabEntry) bool {
    _ = context;

    return std.mem.lessThan(u8, lhs.word, rhs.word);
}

// https://github.com/karpathy/llama2.c/pull/226
// https://github.com/karpathy/llama2.c/pull/297
test "encode utf-8" {
    const tokenizer = try Self.init(std.testing.allocator, "tokenizer.bin", 32000);

    defer tokenizer.deinit();

    const expected = [_]usize{ 365, 1691, 1018, 3963, 669, 29871, 31409, 30607, 30437, 30564 };

    const actual = try tokenizer.encode(
        std.testing.allocator,
        "Lets try ö & 株式会社",
        false,
        false,
    );

    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "encode empty string" {
    const tokenizer = try Self.init(std.testing.allocator, "tokenizer.bin", 32000);

    defer tokenizer.deinit();

    const expected = [_]usize{};

    const actual = try tokenizer.encode(
        std.testing.allocator,
        "",
        false,
        false,
    );

    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "encode unknown codepoint" {
    const tokenizer = try Self.init(std.testing.allocator, "tokenizer.bin", 32000);

    defer tokenizer.deinit();

    const expected = [_]usize{ 29871, 243, 149, 145, 154, 243, 150, 147, 144 };

    const actual = try tokenizer.encode(
        std.testing.allocator,
        "𒎗𓐍",
        false,
        false,
    );

    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "encode single chars" {
    const tokenizer = try Self.init(std.testing.allocator, "tok512.bin", 512);

    defer tokenizer.deinit();

    const expected = [_]usize{ 261, 430, 429, 418, 411, 431, 428, 415 };

    const actual = try tokenizer.encode(
        std.testing.allocator,
        "abcdefgh",
        false,
        false,
    );

    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

// https://github.com/facebookresearch/llama/blob/ea9f33d6d3ea8ed7d560d270986407fd6c2e52b7/example_text_completion.py
test "meta encoding example 1" {
    const tokenizer = try Self.init(std.testing.allocator, "tokenizer.bin", 32000);

    defer tokenizer.deinit();

    const expected = [_]usize{ 1, 306, 4658, 278, 6593, 310, 2834, 338 };

    const actual = try tokenizer.encode(
        std.testing.allocator,
        "I believe the meaning of life is",
        true,
        false,
    );

    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "meta encoding example 2" {
    const tokenizer = try Self.init(std.testing.allocator, "tokenizer.bin", 32000);

    defer tokenizer.deinit();

    const expected = [_]usize{ 1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871, 2 };

    const actual = try tokenizer.encode(
        std.testing.allocator,
        "Simply put, the theory of relativity states that ",
        true,
        true,
    );

    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "meta encoding example 3" {
    const tokenizer = try Self.init(std.testing.allocator, "tokenizer.bin", 32000);

    defer tokenizer.deinit();

    const expected = [_]usize{ 1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871 };

    const actual = try tokenizer.encode(
        std.testing.allocator,
        "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ",
        true,
        false,
    );

    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}

test "meta encoding example 4" {
    const tokenizer = try Self.init(std.testing.allocator, "tokenizer.bin", 32000);

    defer tokenizer.deinit();

    const expected = [_]usize{ 1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149 };

    const actual = try tokenizer.encode(
        std.testing.allocator,
        "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>",
        true,
        false,
    );

    defer std.testing.allocator.free(actual);

    try std.testing.expectEqualSlices(usize, expected[0..], actual);
}
