const std = @import("std");

pub fn print(word: []const u8) !void {
    const stdout = std.io.getStdOut().writer();

    // https://github.com/karpathy/llama2.c/blob/c7a26264a233c32f396b1c67be4ac019d2d8a659/run.c#L427
    if (word.len == 6 and std.mem.eql(u8, word[0..3], "<0x") and word[5] == '>') {
        const byte: ?u8 = std.fmt.parseInt(u8, word[3..5], 16) catch null;

        if (byte) |char| {
            if (std.ascii.isPrint(char) or std.ascii.isWhitespace(char)) {
                try stdout.print("{s}", .{[_]u8{char}});
            }
        } else {
            try stdout.print("{s}", .{word});
        }
    } else {
        try stdout.print("{s}", .{word});
    }
}
