pub fn random(state: *u64) f32 {
    @setFloatMode(.Optimized);

    return @as(f32, @floatFromInt(xorshift(state) >> 8)) / 16777216;
}

// https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
fn xorshift(state: *u64) u32 {
    state.* ^= state.* >> 12;
    state.* ^= state.* << 25;
    state.* ^= state.* >> 27;

    return @intCast((state.* *% 0x2545F4914F6CDD1D) >> 32);
}
