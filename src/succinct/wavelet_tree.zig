//! Wavelet Tree — succinct data structure extending rank/select to arbitrary alphabets.
//!
//! A wavelet tree represents a sequence of length `n` over an alphabet of size `σ`
//! using `n ⌈log₂ σ⌉` bits plus O(n log σ / log n) bits of overhead.
//! It supports rank_c, select_c, and access queries in O(log σ) time.
//!
//! `WaveletTree(SymbolType, IndexType)` is generic over both:
//!  - `SymbolType` — controls alphabet width (u8 for bytes, u16 for Unicode BMP, etc.)
//!  - `IndexType` — controls position/length width (u16 for tiny, u32 for normal, u64 for huge)
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
//! [sizeof(IndexType) bytes: n]            — sequence length (LE)
//! [1 byte: sigma_bits]                    — ⌈log₂ σ⌉ (depth of tree)
//! [sizeof(SymbolType) bytes: sigma - 1]   — alphabet size minus 1 (LE)
//! [sigma_bits levels of RankSelect data]
//! ```
//!
//! Each level stores one RankSelect bitvector of `n` bits (one bit per
//! character at that tree level). Level 0 is the root level (MSB of
//! character values), level `sigma_bits - 1` is the leaf level (LSB).
//!
//! The levels use fixed-size layout: [sizeof(IndexType) bytes: rs_data_len] [rs_data_len bytes: RankSelect data].
//!
//! ## References
//!
//! - Grossi, Gupta, Vitter (2003), "High-order entropy-compressed text indexes"
//! - Navarro (2014), "Wavelet Trees for All" (survey)
//! - Claude & Navarro (2008), "Practical Rank/Select Queries over Arbitrary Sequences"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;
const RankSelect = @import("../succinct/rank_select.zig").RankSelect;

pub fn WaveletTree(comptime SymbolType: type, comptime IndexType: type) type {
    comptime {
        const sym_info = @typeInfo(SymbolType);
        if (sym_info != .int or sym_info.int.signedness != .unsigned)
            @compileError("WaveletTree: SymbolType must be an unsigned integer type, got " ++ @typeName(SymbolType));
        if (@bitSizeOf(SymbolType) < 1)
            @compileError("WaveletTree: SymbolType must be at least 1 bit wide");
        if (@bitSizeOf(SymbolType) > 16)
            @compileError("WaveletTree: SymbolType wider than u16 is not supported");

        const idx_info = @typeInfo(IndexType);
        if (idx_info != .int or idx_info.int.signedness != .unsigned)
            @compileError("WaveletTree: IndexType must be an unsigned integer type, got " ++ @typeName(IndexType));
        if (@sizeOf(IndexType) < 2)
            @compileError("WaveletTree: IndexType must be at least u16");
    }

    const RS = RankSelect(IndexType);
    const symbol_bits = @bitSizeOf(SymbolType);
    const index_size = @sizeOf(IndexType);
    const symbol_size = @sizeOf(SymbolType);

    // Byte-aligned type used to serialize SymbolType values in the header.
    // e.g. u4 → u8, u8 → u8, u9 → u16, u16 → u16.
    const SymbolStorageType = std.meta.Int(.unsigned, symbol_size * 8);

    // Type for bit-shift amounts: must hold values 0..symbol_bits-1
    const ShiftType = std.math.Log2Int(SymbolType);

    // Sigma (alphabet size) stored as a type that can hold symbol_count.
    // For u8 symbols, max sigma = 256, needs u16. For u16, max sigma = 65536, needs u32.
    const SigmaCountType = std.meta.Int(.unsigned, symbol_bits + 1);

    return struct {
        const Self = @This();

        pub const Error = error{
            EmptyInput,
            InvalidData,
            OutOfMemory,
        };

        /// Byte size of the serialized header: n + sigma_bits + (sigma - 1).
        /// Sigma is stored as `alpha_size - 1` in SymbolType to avoid overflow
        /// when the full alphabet is in use (e.g. 256 symbols in u8).
        pub const HEADER_SIZE: usize = index_size + 1 + symbol_size;

        // ── Build-time ──

        /// Build a wavelet tree from a symbol sequence using ArenaConfig.
        /// Temp allocations use `arena.temp`, output from `arena.output`.
        pub fn build(arena: ArenaConfig, input: []const SymbolType) ![]u8 {
            return buildImpl(arena.temp, arena.output, input);
        }

        /// Build with a single allocator.
        pub fn buildUniform(allocator: std.mem.Allocator, input: []const SymbolType) ![]u8 {
            return buildImpl(allocator, allocator, input);
        }

        fn buildImpl(
            temp: std.mem.Allocator,
            output: std.mem.Allocator,
            input: []const SymbolType,
        ) ![]u8 {
            if (input.len == 0) return Error.EmptyInput;

            const n: IndexType = @intCast(input.len);

            // Determine alphabet range
            var max_val: SymbolType = 0;
            for (input) |c| {
                if (c > max_val) max_val = c;
            }
            const alpha_size: SigmaCountType = @as(SigmaCountType, max_val) + 1;
            const sigma_bits_val: u8 = if (alpha_size <= 1) 1 else @intCast(std.math.log2_int(SigmaCountType, alpha_size - 1) + 1);

            // Current permutation of indices (sorted by prefix so far)
            var perm = try temp.alloc(IndexType, n);
            defer temp.free(perm);
            for (0..n) |i| perm[i] = @intCast(i);

            // Temp bitvector for each level
            var bits = try temp.alloc(u1, n);
            defer temp.free(bits);

            // Build RS data for each level into a temp buffer
            var level_data = std.ArrayListUnmanaged(u8){};
            defer level_data.deinit(temp);

            // Temp for the stable partition
            var perm_tmp = try temp.alloc(IndexType, n);
            defer temp.free(perm_tmp);

            for (0..sigma_bits_val) |level| {
                const bit_shift: ShiftType = @intCast(sigma_bits_val - 1 - level);

                // For each position in current permutation, extract the relevant bit
                for (0..n) |i| {
                    const c = input[perm[i]];
                    bits[i] = @intCast((c >> bit_shift) & 1);
                }

                // Build RankSelect for this level's bitvector
                const rs_data = try RS.build(temp, bits);
                defer temp.free(rs_data);

                // Store level: [IndexType: rs_len] [rs_data]
                const rs_len: IndexType = @intCast(rs_data.len);
                var len_buf: [index_size]u8 = undefined;
                std.mem.writeInt(IndexType, &len_buf, rs_len, .little);
                try level_data.appendSlice(temp, &len_buf);
                try level_data.appendSlice(temp, rs_data);

                // Stable partition: 0-bits go left, 1-bits go right.
                var left: IndexType = 0;
                for (0..n) |i| {
                    if (bits[i] == 0) {
                        left += 1;
                    }
                }

                var l_pos: IndexType = 0;
                var r_pos: IndexType = left;
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
            const total_size = HEADER_SIZE + level_data.items.len;
            const result = try output.alloc(u8, total_size);

            // Header: [IndexType: n] [u8: sigma_bits] [SymbolStorageType: sigma - 1]
            std.mem.writeInt(IndexType, result[0..index_size], n, .little);
            result[index_size] = sigma_bits_val;
            std.mem.writeInt(SymbolStorageType, result[index_size + 1 ..][0..symbol_size], @intCast(alpha_size - 1), .little);

            // Level data
            @memcpy(result[HEADER_SIZE..][0..level_data.items.len], level_data.items);

            return result;
        }

        // ── Read-time queries ──

        /// Validate a serialized wavelet tree blob. Call this before querying
        /// if the data comes from an untrusted source (network, disk, etc.).
        /// Returns `InvalidData` if the header is malformed or level data
        /// lengths exceed the blob.
        pub fn validate(data: []const u8) Error!void {
            if (data.len < HEADER_SIZE) return Error.InvalidData;

            const n = std.mem.readInt(IndexType, data[0..index_size], .little);
            const sigma_bits_val = data[index_size];
            if (sigma_bits_val == 0) return Error.InvalidData;
            if (sigma_bits_val > symbol_bits) return Error.InvalidData;

            // Walk level headers to verify all rs_len values are in bounds
            // and each RankSelect payload is internally consistent.
            var off: usize = HEADER_SIZE;
            for (0..sigma_bits_val) |_| {
                if (off + index_size > data.len) return Error.InvalidData;
                const rs_len = std.mem.readInt(IndexType, data[off..][0..index_size], .little);
                off += index_size;
                if (rs_len > data.len - off) return Error.InvalidData;
                const rs_data = data[off..][0..rs_len];
                RS.validate(rs_data) catch return Error.InvalidData;
                if (RS.totalBits(rs_data) != n) return Error.InvalidData;
                off += rs_len;
            }
        }

        /// Sequence length.
        pub fn length(data: []const u8) IndexType {
            return std.mem.readInt(IndexType, data[0..index_size], .little);
        }

        /// Number of distinct bits per character (tree depth).
        pub fn depth(data: []const u8) u8 {
            return data[index_size];
        }

        /// Alphabet size.
        pub fn sigma(data: []const u8) SigmaCountType {
            const raw = std.mem.readInt(SymbolStorageType, data[index_size + 1 ..][0..symbol_size], .little);
            return @as(SigmaCountType, raw) + 1;
        }

        /// Access: get the character at position `pos`.
        pub fn access(data: []const u8, pos: IndexType) SymbolType {
            const n = std.mem.readInt(IndexType, data[0..index_size], .little);
            const sigma_bits_val = data[index_size];

            var c: SymbolType = 0;
            var p: IndexType = pos;
            var level_offset: usize = HEADER_SIZE;

            for (0..sigma_bits_val) |level| {
                const rs_len = std.mem.readInt(IndexType, data[level_offset..][0..index_size], .little);
                const rs_data = data[level_offset + index_size ..][0..rs_len];
                level_offset += index_size + rs_len;

                const bit = RS.getBit(rs_data, p);
                const shift: ShiftType = @intCast(sigma_bits_val - 1 - level);
                c |= @as(SymbolType, bit) << shift;

                // Navigate: bit=0 → go left (rank0), bit=1 → go right (rank1).
                // Left child contains rank0(n) elements; right child starts after them.
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
        pub fn rank(data: []const u8, c: SymbolType, pos: IndexType) IndexType {
            const n = std.mem.readInt(IndexType, data[0..index_size], .little);
            const sigma_bits_val = data[index_size];

            var lo: IndexType = 0;
            var hi: IndexType = pos;
            var level_offset: usize = HEADER_SIZE;

            for (0..sigma_bits_val) |level| {
                const rs_len = std.mem.readInt(IndexType, data[level_offset..][0..index_size], .little);
                const rs_data = data[level_offset + index_size ..][0..rs_len];
                level_offset += index_size + rs_len;

                const shift: ShiftType = @intCast(sigma_bits_val - 1 - level);
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
        pub fn select(data: []const u8, c: SymbolType, i: IndexType) IndexType {
            const n: IndexType = length(data);

            // Strategy: binary search using rank.
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
            if (lo < n and rank(data, c, lo + 1) == i + 1) {
                return lo;
            }
            return n;
        }

        /// Count occurrences of character `c` in the entire sequence.
        pub fn countChar(data: []const u8, c: SymbolType) IndexType {
            const n: IndexType = length(data);
            return rank(data, c, n);
        }

        // ── Internal helper ──

        /// Get the byte offset to level `level` in the serialized data.
        fn levelOffset(data: []const u8, level: usize) usize {
            var off: usize = HEADER_SIZE;
            for (0..level) |_| {
                const rs_len = std.mem.readInt(IndexType, data[off..][0..index_size], .little);
                off += index_size + rs_len;
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
    const WT = WaveletTree(u8, u32);
    const input = "abracadabra";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    for (input, 0..) |c, i| {
        try testing.expectEqual(c, WT.access(data, @intCast(i)));
    }
}

test "WaveletTree: rank" {
    const WT = WaveletTree(u8, u32);
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
    const WT = WaveletTree(u8, u32);
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
    const WT = WaveletTree(u8, u32);
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
    const WT = WaveletTree(u8, u32);
    const input = "aaaaaa";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 6), WT.countChar(data, 'a'));
    for (0..6) |i| {
        try testing.expectEqual(@as(u8, 'a'), WT.access(data, @intCast(i)));
    }
}

test "WaveletTree: binary alphabet" {
    const WT = WaveletTree(u8, u32);
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
    const WT = WaveletTree(u8, u32);
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
    const WT = WaveletTree(u8, u32);
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
    const WT = WaveletTree(u8, u32);
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
    const WT = WaveletTree(u8, u32);
    const arena = ArenaConfig.uniform(testing.allocator);
    const data = try WT.build(arena, "test");
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u8, 't'), WT.access(data, 0));
    try testing.expectEqual(@as(u8, 'e'), WT.access(data, 1));
}

test "WaveletTree: u16 symbols" {
    const WT = WaveletTree(u16, u32);
    // Simulate a small Unicode-like sequence with codepoints > 255
    const input = [_]u16{ 0x0041, 0x00E9, 0x4E16, 0x754C, 0x0041, 0x4E16 };
    const data = try WT.buildUniform(testing.allocator, &input);
    defer testing.allocator.free(data);

    // Access round-trip
    for (input, 0..) |c, i| {
        try testing.expectEqual(c, WT.access(data, @intCast(i)));
    }

    // Rank: 'A' (0x0041) appears at positions 0 and 4
    try testing.expectEqual(@as(u32, 0), WT.rank(data, 0x0041, 0));
    try testing.expectEqual(@as(u32, 1), WT.rank(data, 0x0041, 1));
    try testing.expectEqual(@as(u32, 2), WT.rank(data, 0x0041, 6));

    // Count: 0x4E16 appears twice
    try testing.expectEqual(@as(u32, 2), WT.countChar(data, 0x4E16));

    // Select: first occurrence of 0x00E9
    try testing.expectEqual(@as(u32, 1), WT.select(data, 0x00E9, 0));
}

test "WaveletTree: u16 IndexType" {
    const WT = WaveletTree(u8, u16);
    const input = "abracadabra";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    for (input, 0..) |c, i| {
        try testing.expectEqual(c, WT.access(data, @intCast(i)));
    }
    try testing.expectEqual(@as(u16, 5), WT.countChar(data, 'a'));
    try testing.expectEqual(@as(u16, 0), WT.select(data, 'a', 0));
}

test "WaveletTree: u4 symbols (tiny alphabet)" {
    const WT = WaveletTree(u4, u32);
    const input = [_]u4{ 0, 1, 2, 3, 4, 5, 0, 1, 2, 3 };
    const data = try WT.buildUniform(testing.allocator, &input);
    defer testing.allocator.free(data);

    for (input, 0..) |c, i| {
        try testing.expectEqual(c, WT.access(data, @intCast(i)));
    }
    try testing.expectEqual(@as(u32, 2), WT.countChar(data, 0));
    try testing.expectEqual(@as(u32, 2), WT.countChar(data, 3));
    try testing.expectEqual(@as(u32, 1), WT.countChar(data, 5));
}

test "WaveletTree: empty input returns error" {
    const WT = WaveletTree(u8, u32);
    try testing.expectError(error.EmptyInput, WT.buildUniform(testing.allocator, ""));
}

test "WaveletTree: u1 binary alphabet (single RankSelect level)" {
    // u1 is the degenerate case: alphabet {0, 1}, tree depth = 1.
    // Equivalent to a single RankSelect bitvector.
    const WT = WaveletTree(u1, u32);
    const input = [_]u1{ 0, 1, 0, 1, 1, 0, 0, 1 };
    const data = try WT.buildUniform(testing.allocator, &input);
    defer testing.allocator.free(data);

    // Access round-trip
    for (input, 0..) |c, i| {
        try testing.expectEqual(c, WT.access(data, @intCast(i)));
    }

    // Counts
    try testing.expectEqual(@as(u32, 4), WT.countChar(data, 0));
    try testing.expectEqual(@as(u32, 4), WT.countChar(data, 1));

    // Rank
    try testing.expectEqual(@as(u32, 0), WT.rank(data, 1, 0));
    try testing.expectEqual(@as(u32, 1), WT.rank(data, 1, 2));
    try testing.expectEqual(@as(u32, 4), WT.rank(data, 1, 8));

    // Select
    try testing.expectEqual(@as(u32, 1), WT.select(data, 1, 0)); // first '1' at pos 1
    try testing.expectEqual(@as(u32, 3), WT.select(data, 1, 1)); // second '1' at pos 3
}

test "WaveletTree: validate rejects corrupt data" {
    const WT = WaveletTree(u8, u32);

    // Valid blob first — build then validate succeeds.
    const data = try WT.buildUniform(testing.allocator, "abracadabra");
    defer testing.allocator.free(data);
    try WT.validate(data);

    // Blob too short for header.
    try testing.expectError(error.InvalidData, WT.validate(&[_]u8{ 0, 1, 2 }));
    try testing.expectError(error.InvalidData, WT.validate(&[_]u8{}));

    // sigma_bits > 8 (impossible for u8 SymbolType).
    var bad = try testing.allocator.dupe(u8, data);
    defer testing.allocator.free(bad);
    bad[4] = 12; // sigma_bits byte, should be <= 8
    try testing.expectError(error.InvalidData, WT.validate(bad));

    // sigma_bits == 0 — build never produces this; queries would silently return bogus.
    var zero_depth = try testing.allocator.dupe(u8, data);
    defer testing.allocator.free(zero_depth);
    zero_depth[4] = 0;
    try testing.expectError(error.InvalidData, WT.validate(zero_depth));

    // Truncated level data — valid header but blob cut short.
    const truncated = data[0 .. WT.HEADER_SIZE + 2];
    try testing.expectError(error.InvalidData, WT.validate(truncated));
}

test "WaveletTree: validate rejects corrupted RankSelect internals" {
    const WT = WaveletTree(u8, u32);
    const original = try WT.buildUniform(testing.allocator, "abracadabra");
    var data = try testing.allocator.dupe(u8, original);
    testing.allocator.free(original);
    defer testing.allocator.free(data);

    // Find first RS level and inflate its total_bits to cause OOB.
    const rs_offset = WT.HEADER_SIZE + @sizeOf(u32); // skip rs_len header
    std.mem.writeInt(u32, data[rs_offset..][0..4], 999999, .little);
    try testing.expectError(error.InvalidData, WT.validate(data));
}

test "WaveletTree: validate rejects RS bit count mismatch" {
    const WT = WaveletTree(u8, u32);
    const original = try WT.buildUniform(testing.allocator, "abracadabra");
    var data = try testing.allocator.dupe(u8, original);
    defer testing.allocator.free(data);
    testing.allocator.free(original);

    // Corrupt the RS total_bits to n+1 — should fail the RS.totalBits != n check.
    // n = 11 for "abracadabra", so set RS total_bits to 12.
    const rs_offset = WT.HEADER_SIZE + @sizeOf(u32); // skip rs_len header
    std.mem.writeInt(u32, data[rs_offset..][0..4], 12, .little);
    try testing.expectError(error.InvalidData, WT.validate(data));
}
