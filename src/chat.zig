const Self = @This();

const std = @import("std");
const ChatArgs = @import("chat_args.zig");
const print = @import("print.zig").print;
const Sampler = @import("sampler.zig");
const Tokenizer = @import("tokenizer.zig");
const Transformer = @import("transformer.zig");

allocator: std.mem.Allocator,
transformer: Transformer,
tokenizer: Tokenizer,
sampler: Sampler,
system_prompt: []const u8,
user_prompt: []const u8,

pub fn init(allocator: std.mem.Allocator, args: ChatArgs) !Self {
    const transformer = try Transformer.init(allocator, args.model_path, args.sequence_length);

    errdefer transformer.deinit();

    const vocab_size = transformer.checkpoint.vocab_size;
    const tokenizer = try Tokenizer.init(allocator, args.model_path, vocab_size);

    errdefer tokenizer.deinit();

    const sampler = try Sampler.init(allocator, args, vocab_size);

    errdefer sampler.deinit();

    return Self{
        .allocator = allocator,
        .transformer = transformer,
        .tokenizer = tokenizer,
        .sampler = sampler,
        .system_prompt = args.system_prompt,
        .user_prompt = args.user_prompt,
    };
}

pub fn deinit(self: *const Self) void {
    self.transformer.deinit();
    self.tokenizer.deinit();
    self.sampler.deinit();
}

const system_prompt_template_start = "<<SYS>>\n";
const system_prompt_template_close = "\n<</SYS>>\n\n";
const user_prompt_template_start = "[INST] ";
const user_prompt_template_close = " [/INST]";

const bos_token = 1; // beginning of sequence
const eos_token = 2; // end of sequence

pub fn start(self: *Self, allocator: std.mem.Allocator) !void {
    var stdin = std.io.getStdIn().reader();
    var stdout = std.io.getStdOut().writer();

    var token: usize = bos_token;
    var next_token: usize = 0;
    var user_turn: bool = true;
    var user_prompt_tokens_index: usize = 0;

    var user_prompt_tokens: ?[]const usize = null;

    defer if (user_prompt_tokens) |prompt_tokens| {
        allocator.free(prompt_tokens);
    };

    for (0..self.transformer.sequence_length) |position| {
        self.transformer.forward(token, position);

        if (token == bos_token and user_turn) {
            var user_prompt = std.ArrayList(u8).init(allocator);

            defer user_prompt.deinit();

            try user_prompt.appendSlice(user_prompt_template_start);

            if (position == 0) {
                if (self.system_prompt.len == 0) {
                    var system_prompt = std.ArrayList(u8).init(allocator);

                    defer system_prompt.deinit();

                    try stdout.print("Enter system prompt (optional): ", .{});
                    try stdin.streamUntilDelimiter(system_prompt.writer(), '\n', null);

                    if (system_prompt.items.len > 0) {
                        try user_prompt.appendSlice(system_prompt_template_start);
                        try user_prompt.appendSlice(try system_prompt.toOwnedSlice());
                        try user_prompt.appendSlice(system_prompt_template_close);
                    }
                } else {
                    try user_prompt.appendSlice(system_prompt_template_start);
                    try user_prompt.appendSlice(self.system_prompt);
                    try user_prompt.appendSlice(system_prompt_template_close);
                }
            }

            if (position == 0 and self.user_prompt.len > 0) {
                try user_prompt.appendSlice(self.user_prompt);
            } else {
                try stdout.print("User: ", .{});
                try stdin.streamUntilDelimiter(user_prompt.writer(), '\n', null);
            }

            try user_prompt.appendSlice(user_prompt_template_close);

            if (user_prompt_tokens) |prompt_tokens| {
                allocator.free(prompt_tokens);

                user_prompt_tokens = null;
            }

            user_turn = false;
            user_prompt_tokens_index = 0;
            user_prompt_tokens = try self.tokenizer.encode(allocator, user_prompt.items);

            try stdout.print("Assistant:", .{});
        }

        if (user_prompt_tokens) |prompt_tokens| {
            if (user_prompt_tokens_index < prompt_tokens.len) {
                next_token = prompt_tokens[user_prompt_tokens_index];
            }
        }

        user_prompt_tokens_index += 1;

        if (next_token == 0) {
            next_token = self.sampler.sample(self.transformer.output_buffer.values);
        }

        if (next_token == eos_token) {
            user_turn = true;

            try stdout.print("\n", .{});
        } else if (user_prompt_tokens) |prompt_tokens| {
            if (next_token > 2 and user_prompt_tokens_index > prompt_tokens.len) {
                const word = self.tokenizer.decode(
                    next_token,
                    user_prompt_tokens_index == prompt_tokens.len + 1,
                );

                try print(word, stdout);
            }
        }

        token = next_token;
        next_token = 0;
    }

    try stdout.print("\n", .{});
}
