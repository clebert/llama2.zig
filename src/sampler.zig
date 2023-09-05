const Self = @This();

const std = @import("std");
const lib = @import("lib.zig");
const Cli = @import("cli.zig");

allocator: std.mem.Allocator,
probability_index_pairs_buffer: []lib.ProbabilityIndexPair,
temperature: f32,
top_p: f32,
rng_state: u64,

pub fn init(allocator: std.mem.Allocator, cli: *const Cli, vocab_size: usize) !Self {
    const probability_index_pairs_buffer =
        try allocator.alloc(lib.ProbabilityIndexPair, vocab_size);

    return Self{
        .allocator = allocator,
        .probability_index_pairs_buffer = probability_index_pairs_buffer,
        .temperature = cli.temperature,
        .top_p = cli.top_p,
        .rng_state = cli.random_seed,
    };
}

pub fn deinit(self: *const Self) void {
    self.allocator.free(self.probability_index_pairs_buffer);
}

pub fn sample(self: *Self, probability_distribution: []f32) usize {
    if (self.temperature == 0) {
        return lib.argmax(probability_distribution);
    }

    for (probability_distribution) |*probability| {
        probability.* /= self.temperature;
    }

    lib.softmax(probability_distribution);

    if (self.top_p <= 0 or self.top_p >= 1) {
        return lib.sampleMultinomial(lib.random(&self.rng_state), probability_distribution);
    }

    return lib.sampleNucleus(
        lib.random(&self.rng_state),
        probability_distribution,
        self.top_p,
        self.probability_index_pairs_buffer,
    );
}
