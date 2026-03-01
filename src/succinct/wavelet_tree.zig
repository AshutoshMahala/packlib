//! Wavelet Tree — succinct data structure extending rank/select to arbitrary alphabets.
//!
//! A wavelet tree represents a string of length `n` over an alphabet of size `σ`
//! using `n ⌈log₂ σ⌉` bits plus O(n log σ / log n) bits of overhead.
//! It supports rank_c, select_c, and access queries in O(log σ) time.
//!
//! This implementation uses a balanced binary wavelet tree where the
//! alphabet is partitioned by bit positions (most significant bit first).
//! Each internal node stores a RankSelect bitvector that classifies
//! characters as going left (bit=0) or right (bit=1).
//!
//! **Build-time:** allocates (uses `ArenaConfig`).
//! **Read-time:** zero allocation — all queries on `[]const u8`.
//!
//! ## Serialized format
//!
//! ```text
//! [u32: n]                        — sequence length
//! [u8: sigma_bits]                — ⌈log₂ σ⌉ (depth of tree)
//! [u16: sigma]                    — alphabet size
//! [sigma_bits levels of RankSelect data]
//! ```
//!
//! Each level stores one RankSelect bitvector of `n` bits (one bit per
//! character at that tree level). Level 0 is the root level (MSB of
//! character values), level `sigma_bits - 1` is the leaf level (LSB).
//!
//! The levels use fixed-size layout: [u32: rs_data_len] [rs_data_len bytes: RankSelect data].
//!
//! ## References
//!
//! - Grossi, Gupta, Vitter (2003), "High-order entropy-compressed text indexes"
//! - Navarro (2014), "Wavelet Trees for All" (survey)
//! - Claude & Navarro (2008), "Practical Rank/Select Queries over Arbitrary Sequences"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;
const RankSelect = @import("../succinct/rank_select.zig").RankSelect;

pub fn WaveletTree(comptime IndexType: type) type {
    const RS = RankSelect(IndexType);

    return struct {
        const Self = @This();

        pub const Error = error{
            EmptyInput,
            InvalidData,
            OutOfMemory,
        };

        // Header: [u32: n] [u8: sigma_bits] [u16: sigma]
        const HEADER_SIZE: u32 = 4 + 1 + 2;

        // ── Build-time ──

        /// Build a wavelet tree from a byte sequence using ArenaConfig.
        /// Temp allocations use `arena.temp`, output from `arena.output`.
        pub fn build(arena: ArenaConfig, input: []const u8) ![]u8 {
            return buildImpl(arena.temp, arena.output, input);
        }

        /// Build with a single allocator.
        pub fn buildUniform(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
            return buildImpl(allocator, allocator, input);
        }

        fn buildImpl(
            temp: std.mem.Allocator,
            output: std.mem.Allocator,
            input: []const u8,
        ) ![]u8 {
            if (input.len == 0) return Error.EmptyInput;

            const n: u32 = @intCast(input.len);

            // Determine alphabet range
            var max_val: u8 = 0;
            for (input) |c| {
                if (c > max_val) max_val = c;
            }
            const alpha_size: u16 = @as(u16, max_val) + 1;
            const sigma_bits: u8 = if (alpha_size <= 1) 1 else @intCast(std.math.log2_int(u16, alpha_size - 1) + 1);

            // Build each level's bitvector
            // At each level, we classify characters by their bit at that level.
            // We need to track which characters are at each node at each level.

            // Current permutation of indices (sorted by prefix so far)
            var perm = try temp.alloc(u32, n);
            defer temp.free(perm);
            for (0..n) |i| perm[i] = @intCast(i);

            // Temp bitvector for each level
            var bits = try temp.alloc(u1, n);
            defer temp.free(bits);

            // Build RS data for each level into a temp buffer
            var level_data = std.ArrayListUnmanaged(u8){};
            defer level_data.deinit(temp);

            // Temp for the stable partition
            var perm_tmp = try temp.alloc(u32, n);
            defer temp.free(perm_tmp);

            for (0..sigma_bits) |level| {
                const bit_shift: u3 = @intCast(sigma_bits - 1 - level);

                // For each position in current permutation, extract the relevant bit
                for (0..n) |i| {
                    const c = input[perm[i]];
                    bits[i] = @intCast((c >> bit_shift) & 1);
                }

                // Build RankSelect for this level's bitvector
                const rs_data = try RS.build(temp, bits);
                defer temp.free(rs_data);

                // Store level: [u32: rs_len] [rs_data]
                const rs_len: u32 = @intCast(rs_data.len);
                var len_buf: [4]u8 = undefined;
                std.mem.writeInt(u32, &len_buf, rs_len, .little);
                try level_data.appendSlice(temp, &len_buf);
                try level_data.appendSlice(temp, rs_data);

                // Stable partition: 0-bits go left, 1-bits go right.
                // This reorders perm so that the left subtree's elements come first.
                var left: u32 = 0;
                for (0..n) |i| {
                    if (bits[i] == 0) {
                        left += 1;
                    }
                }

                var l_pos: u32 = 0;
                var r_pos: u32 = left;
                for (0..n) |i| {
                    if (bits[i] == 0) {
                        perm_tmp[l_pos] = perm[i];
                        l_pos += 1;
                    } else {
                        perm_tmp[r_pos] = perm[i];
                        r_pos += 1;
                    }
                }
                @memcpy(perm, perm_tmp);
            }

            // Assemble output
            const total_size = HEADER_SIZE + @as(u32, @intCast(level_data.items.len));
            const result = try output.alloc(u8, total_size);

            // Header
            std.mem.writeInt(u32, result[0..4], n, .little);
            result[4] = sigma_bits;
            std.mem.writeInt(u16, result[5..7], alpha_size, .little);

            // Level data
            @memcpy(result[HEADER_SIZE..][0..level_data.items.len], level_data.items);

            return result;
        }

        // ── Read-time queries ──

        /// Sequence length.
        pub fn length(data: []const u8) u32 {
            return std.mem.readInt(u32, data[0..4], .little);
        }

        /// Number of distinct bits per character.
        pub fn depth(data: []const u8) u8 {
            return data[4];
        }

        /// Alphabet size.
        pub fn sigma(data: []const u8) u16 {
            return std.mem.readInt(u16, data[5..7], .little);
        }

        /// Access: get the character at position `pos`.
        pub fn access(data: []const u8, pos: IndexType) u8 {
            const n = std.mem.readInt(u32, data[0..4], .little);
            const sigma_bits = data[4];

            var c: u8 = 0;
            var p: IndexType = pos;
            var level_offset: u32 = HEADER_SIZE;

            for (0..sigma_bits) |level| {
                const rs_len = std.mem.readInt(u32, data[level_offset..][0..4], .little);
                const rs_data = data[level_offset + 4 ..][0..rs_len];
                level_offset += 4 + rs_len;

                const bit = RS.getBit(rs_data, p);
                const shift: u3 = @intCast(sigma_bits - 1 - level);
                c |= @as(u8, bit) << shift;

                // Navigate: if bit=0, go left; if bit=1, go right.
                // Left child contains rank0(n) elements, indexed by rank0.
                // Right child starts after rank0(n) elements, indexed by rank1.
                if (bit == 0) {
                    p = RS.rank0(rs_data, p);
                } else {
                    const zeros = RS.rank0(rs_data, @intCast(n));
                    p = zeros + RS.rank1(rs_data, p);
                }
            }
            return c;
        }

        /// Rank: count occurrences of character `c` in positions [0, pos).
        pub fn rank(data: []const u8, c: u8, pos: IndexType) IndexType {
            const n = std.mem.readInt(u32, data[0..4], .little);
            const sigma_bits = data[4];

            var lo: IndexType = 0;
            var hi: IndexType = pos;
            var level_offset: u32 = HEADER_SIZE;

            for (0..sigma_bits) |level| {
                const rs_len = std.mem.readInt(u32, data[level_offset..][0..4], .little);
                const rs_data = data[level_offset + 4 ..][0..rs_len];
                level_offset += 4 + rs_len;

                const shift: u3 = @intCast(sigma_bits - 1 - level);
                const bit: u1 = @intCast((c >> shift) & 1);

                if (bit == 0) {
                    lo = RS.rank0(rs_data, lo);
                    hi = RS.rank0(rs_data, hi);
                } else {
                    const zeros = RS.rank0(rs_data, @intCast(n));
                    lo = zeros + RS.rank1(rs_data, lo);
                    hi = zeros + RS.rank1(rs_data, hi);
                }
            }

            return hi - lo;
        }

        /// Select: position of the i-th occurrence of character `c` (0-indexed).
        /// Returns `length` if not found.
        pub fn select(data: []const u8, c: u8, i: IndexType) IndexType {
            const n: IndexType = @intCast(std.mem.readInt(u32, data[0..4], .little));

            // Strategy: binary search using rank.
            // Find smallest pos such that rank(c, pos + 1) > i.
            var lo: IndexType = 0;
            var hi: IndexType = n;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (rank(data, c, mid + 1) <= i) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            // Verify it's actually character c at position lo
            if (lo < n and rank(data, c, lo + 1) == i + 1) {
                return lo;
            }
            return n; // not found
        }

        /// Count occurrences of character `c` in the entire sequence.
        pub fn countChar(data: []const u8, c: u8) IndexType {
            const n: IndexType = @intCast(std.mem.readInt(u32, data[0..4], .little));
            return rank(data, c, n);
        }

        // ── Internal helpers for level navigation ──

        /// Get the offset to level `level` in the serialized data.
        fn levelOffset(data: []const u8, level: u32) u32 {
            var off: u32 = HEADER_SIZE;
            for (0..level) |_| {
                const rs_len = std.mem.readInt(u32, data[off..][0..4], .little);
                off += 4 + rs_len;
            }
            return off;
        }
    };
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "WaveletTree: access all positions" {
    const WT = WaveletTree(u32);
    const input = "abracadabra";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    for (input, 0..) |c, i| {
        try testing.expectEqual(c, WT.access(data, @intCast(i)));
    }
}

test "WaveletTree: rank" {
    const WT = WaveletTree(u32);
    const input = "abracadabra";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    // Count 'a' in prefixes:
    // input:  a b r a c a d a b r a
    // rank_a: 0 1 1 1 2 2 3 3 4 4 4 5
    try testing.expectEqual(@as(u32, 0), WT.rank(data, 'a', 0));
    try testing.expectEqual(@as(u32, 1), WT.rank(data, 'a', 1));
    try testing.expectEqual(@as(u32, 1), WT.rank(data, 'a', 2));
    try testing.expectEqual(@as(u32, 2), WT.rank(data, 'a', 4));
    try testing.expectEqual(@as(u32, 5), WT.rank(data, 'a', 11));

    // Count 'b':
    try testing.expectEqual(@as(u32, 0), WT.rank(data, 'b', 0));
    try testing.expectEqual(@as(u32, 0), WT.rank(data, 'b', 1));
    try testing.expectEqual(@as(u32, 1), WT.rank(data, 'b', 2));
    try testing.expectEqual(@as(u32, 2), WT.rank(data, 'b', 11));

    // Count 'r':
    try testing.expectEqual(@as(u32, 2), WT.rank(data, 'r', 11));
}

test "WaveletTree: select" {
    const WT = WaveletTree(u32);
    const input = "abracadabra";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    // select('a', 0) = 0 (first 'a')
    try testing.expectEqual(@as(u32, 0), WT.select(data, 'a', 0));
    // select('a', 1) = 3 (second 'a')
    try testing.expectEqual(@as(u32, 3), WT.select(data, 'a', 1));
    // select('a', 4) = 10 (fifth 'a')
    try testing.expectEqual(@as(u32, 10), WT.select(data, 'a', 4));

    // select('b', 0) = 1
    try testing.expectEqual(@as(u32, 1), WT.select(data, 'b', 0));
    // select('b', 1) = 8
    try testing.expectEqual(@as(u32, 8), WT.select(data, 'b', 1));

    // select('z', 0) = n (not found)
    try testing.expectEqual(@as(u32, 11), WT.select(data, 'z', 0));
}

test "WaveletTree: countChar" {
    const WT = WaveletTree(u32);
    const input = "abracadabra";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 5), WT.countChar(data, 'a'));
    try testing.expectEqual(@as(u32, 2), WT.countChar(data, 'b'));
    try testing.expectEqual(@as(u32, 2), WT.countChar(data, 'r'));
    try testing.expectEqual(@as(u32, 1), WT.countChar(data, 'c'));
    try testing.expectEqual(@as(u32, 1), WT.countChar(data, 'd'));
    try testing.expectEqual(@as(u32, 0), WT.countChar(data, 'z'));
}

test "WaveletTree: single character repeated" {
    const WT = WaveletTree(u32);
    const input = "aaaaaa";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 6), WT.countChar(data, 'a'));
    for (0..6) |i| {
        try testing.expectEqual(@as(u8, 'a'), WT.access(data, @intCast(i)));
    }
}

test "WaveletTree: binary alphabet" {
    const WT = WaveletTree(u32);
    const input = [_]u8{ 0, 1, 0, 1, 1, 0 };
    const data = try WT.buildUniform(testing.allocator, &input);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 3), WT.countChar(data, 0));
    try testing.expectEqual(@as(u32, 3), WT.countChar(data, 1));

    for (input, 0..) |c, i| {
        try testing.expectEqual(c, WT.access(data, @intCast(i)));
    }
}

test "WaveletTree: all byte values" {
    const WT = WaveletTree(u32);
    var input: [256]u8 = undefined;
    for (0..256) |i| input[i] = @intCast(i);

    const data = try WT.buildUniform(testing.allocator, &input);
    defer testing.allocator.free(data);

    for (0..256) |i| {
        try testing.expectEqual(@as(u8, @intCast(i)), WT.access(data, @intCast(i)));
        try testing.expectEqual(@as(u32, 1), WT.countChar(data, @intCast(i)));
    }
}

test "WaveletTree: rank consistency with countChar" {
    const WT = WaveletTree(u32);
    const input = "the quick brown fox jumps over the lazy dog";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    // rank(c, n) should equal countChar(c) for any character
    const n: u32 = @intCast(input.len);
    for (0..128) |c| {
        try testing.expectEqual(WT.countChar(data, @intCast(c)), WT.rank(data, @intCast(c), n));
    }
}

test "WaveletTree: select finds correct positions" {
    const WT = WaveletTree(u32);
    const input = "mississippi";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    // Verify select is consistent with access
    for (0..128) |c_usize| {
        const c: u8 = @intCast(c_usize);
        const total = WT.countChar(data, c);
        for (0..total) |i| {
            const pos = WT.select(data, c, @intCast(i));
            try testing.expectEqual(c, WT.access(data, pos));
        }
    }
}

test "WaveletTree: ArenaConfig variant" {
    const WT = WaveletTree(u32);
    const arena = ArenaConfig.uniform(testing.allocator);
    const data = try WT.build(arena, "test");
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u8, 't'), WT.access(data, 0));
    try testing.expectEqual(@as(u8, 'e'), WT.access(data, 1));
}

test "WaveletTree: empty input returns error" {
    const WT = WaveletTree(u32);
    try testing.expectError(error.EmptyInput, WT.buildUniform(testing.allocator, ""));
}
