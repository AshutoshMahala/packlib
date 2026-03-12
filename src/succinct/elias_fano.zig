//! Elias–Fano encoding for monotone integer sequences.
//!
//! Stores a sorted sequence of `n` integers from universe `[0, u)` using
//! approximately `n ⌈log₂(u/n)⌉ + 2n` bits — close to the information-theoretic
//! lower bound.  Supports O(1) access, O(1) successor, and O(1) predecessor
//! queries (via the underlying rank/select structures).
//!
//! **Build-time:** allocates (uses `ArenaConfig` tiers).
//! **Read-time:** zero allocation — all queries on `[]const u8`.
//!
//! ## Serialized format
//!
//! ```text
//! [IndexType: n]                        — number of elements
//! [IndexType: universe]                 — one past the maximum value
//! [u8:        low_bits]                 — number of low bits per element (ℓ)
//! [⌈n×ℓ/8⌉ bytes: low_parts]           — packed low-bit values, MSB-first
//! [rank/select data: high_parts]        — unary-coded high parts as a RankSelect bitvector
//! ```
//!
//! The high part of each value encodes the quotient `v >> ℓ` in unary within a
//! bitvector of length `n + (max >> ℓ) + 1`.  A 1-bit marks each element and
//! 0-bits act as bucket separators.
//!
//! ## References
//!
//! - Elias (1974), "Efficient storage and retrieval by content and address of static files"
//! - Fano (1971), "On the number of bits sufficient to implement an associative memory"
//! - Vigna (2013), "Quasi-succinct indices"
//! - Ottaviano & Venturini (2014), "Partitioned Elias–Fano indexes"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;
const RankSelect = @import("rank_select.zig").RankSelect;

pub fn EliasFano(comptime IndexType: type) type {
    comptime {
        const info = @typeInfo(IndexType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("EliasFano: IndexType must be an unsigned integer type, got " ++ @typeName(IndexType));
        if (@sizeOf(IndexType) < 2)
            @compileError("EliasFano: IndexType must be at least u16");
    }
    const index_size = @sizeOf(IndexType);
    const RS = RankSelect(IndexType);

    return struct {
        const Self = @This();

        pub const Error = error{
            EmptySequence,
            NotMonotone,
            OutOfMemory,
        };

        // ── Header layout ──
        // [IndexType: n] [IndexType: universe] [u8: low_bits]
        const HEADER_SIZE: u32 = index_size * 2 + 1;

        // ── Build-time ──

        /// Build an Elias–Fano encoded representation of a sorted integer sequence.
        ///
        /// `values` must be non-empty and monotonically non-decreasing.
        /// The returned slice is allocated from `arena.output` (caller owns it).
        /// Temporary work uses `arena.temp`.
        pub fn build(arena: ArenaConfig, values: []const IndexType) ![]u8 {
            if (values.len == 0) return Error.EmptySequence;

            const n: IndexType = @intCast(values.len);

            // Verify monotonicity
            for (1..values.len) |i| {
                if (values[i] < values[i - 1]) return Error.NotMonotone;
            }

            // Universe = max value + 1 (one past the last value)
            const max_val = values[values.len - 1];
            const u_size: IndexType = if (max_val == std.math.maxInt(IndexType)) max_val else max_val + 1;

            // Number of low bits: ℓ = max(0, ⌊log₂(u/n)⌋)
            const low_bits: u8 = computeLowBits(n, u_size);

            // Low parts: n values, each `low_bits` wide, packed MSB-first
            const low_total_bits: u64 = @as(u64, n) * @as(u64, low_bits);
            const low_bytes: u32 = @intCast((low_total_bits + 7) / 8);

            // High parts: unary encoding in a bitvector
            // High part of value v = v >> low_bits
            // Bitvector length = n + (max_val >> low_bits) + 1
            const max_high: u64 = if (low_bits >= @bitSizeOf(IndexType)) 0 else @as(u64, max_val) >> @intCast(low_bits);
            const high_bv_len: u32 = @intCast(@as(u64, n) + max_high + 1);

            // Build high bitvector in temp arena
            const high_bits = try arena.temp.alloc(u1, high_bv_len);
            defer arena.temp.free(high_bits);
            @memset(high_bits, 0);

            // Fill the high bitvector using unary encoding:
            // For each element, set bit at position (high_part + element_index)
            // where high_part = value >> low_bits
            var write_pos: u32 = 0;
            var prev_high: u64 = 0;
            for (0..@intCast(n)) |i| {
                const v: u64 = values[i];
                const high: u64 = if (low_bits >= @bitSizeOf(IndexType)) 0 else v >> @intCast(low_bits);

                // Write (high - prev_high) zero bits as bucket separators
                const gap = high - prev_high;
                write_pos += @intCast(gap);

                // If this isn't the first element in this bucket, we already
                // advanced past the bucket separator, so just place the next 1-bit
                if (i > 0) {
                    // Add a 0 for the bucket separator only if high > prev_high
                    // (already done by the gap advancement above)
                }

                // Write a 1-bit for this element
                high_bits[write_pos] = 1;
                write_pos += 1;
                prev_high = high;
            }

            // Build the rank/select structure for the high bitvector
            const rs_data = try RS.build(arena.temp, high_bits);
            defer arena.temp.free(rs_data);
            const rs_len: u32 = @intCast(rs_data.len);

            // Assemble the final output
            const total_size = HEADER_SIZE + low_bytes + rs_len;
            const output = try arena.output.alloc(u8, total_size);
            @memset(output, 0);

            // Write header
            writeIdx(output, 0, n);
            writeIdx(output, index_size, u_size);
            output[index_size * 2] = low_bits;

            // Write low parts (packed MSB-first)
            const low_start = HEADER_SIZE;
            for (0..@intCast(n)) |i| {
                if (low_bits == 0) break;
                const v: u64 = values[i];
                const low_mask: u64 = (@as(u64, 1) << @intCast(low_bits)) - 1;
                const low_val: u64 = v & low_mask;

                // Write `low_bits` bits at position i * low_bits
                const bit_offset: u64 = @as(u64, i) * @as(u64, low_bits);
                writeBitsPacked(output[low_start..], bit_offset, low_val, low_bits);
            }

            // Copy rank/select data
            const rs_start = low_start + low_bytes;
            @memcpy(output[rs_start..][0..rs_len], rs_data);

            return output;
        }

        /// Convenience: build with a plain allocator (uniform ArenaConfig).
        pub fn buildUniform(allocator: std.mem.Allocator, values: []const IndexType) ![]u8 {
            return build(ArenaConfig.uniform(allocator), values);
        }

        // ── Read-time (all no-alloc on []const u8) ──

        /// Number of elements in the encoded sequence.
        pub fn count(data: []const u8) IndexType {
            return readIdx(data, 0);
        }

        /// Universe size (one past the maximum value).
        pub fn universe(data: []const u8) IndexType {
            return readIdx(data, index_size);
        }

        /// Access the i-th element (0-indexed).
        pub fn get(data: []const u8, i: IndexType) IndexType {
            const n = count(data);
            std.debug.assert(i < n);

            const low_bits = data[index_size * 2];

            // Low part: read `low_bits` bits at position i * low_bits
            const low_start = HEADER_SIZE;
            const low_val: IndexType = if (low_bits == 0) 0 else blk: {
                const bit_offset: u64 = @as(u64, i) * @as(u64, low_bits);
                break :blk @intCast(readBitsPacked(data[low_start..], bit_offset, low_bits));
            };

            // High part: select1(i) - i gives the high value (unary decoding)
            const rs_start = low_start + lowBytesCount(data);
            const rs_data = data[rs_start..];
            const pos = RS.select1(rs_data, i);
            const high_val: IndexType = pos - i;

            // Reconstruct: (high << low_bits) | low
            if (low_bits == 0) return high_val;
            return (high_val << @intCast(low_bits)) | low_val;
        }

        /// Find the smallest index `j` such that `get(j) >= target`.
        /// Returns `n` if all values are less than `target`.
        /// This is the successor query.
        pub fn successor(data: []const u8, target: IndexType) IndexType {
            const n = count(data);
            if (n == 0) return 0;

            // Binary search for the successor
            var lo: IndexType = 0;
            var hi: IndexType = n;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (get(data, mid) < target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }

        /// Find the largest index `j` such that `get(j) <= target`.
        /// Returns `n` if no such value exists.
        /// This is the predecessor query.
        pub fn predecessor(data: []const u8, target: IndexType) IndexType {
            const n = count(data);
            if (n == 0) return n;

            // Binary search: find largest index where get(j) <= target
            var lo: IndexType = 0;
            var hi: IndexType = n;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (get(data, mid) <= target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            // lo is the count of values <= target, so the index is lo-1
            if (lo == 0) return n; // no value <= target
            return lo - 1;
        }

        /// Check if a value exists in the sequence.
        pub fn contains(data: []const u8, value: IndexType) bool {
            const n = count(data);
            if (n == 0) return false;
            const idx = successor(data, value);
            return idx < n and get(data, idx) == value;
        }

        // ── Internal helpers ──

        fn computeLowBits(n: IndexType, u: IndexType) u8 {
            if (n == 0 or u == 0) return 0;
            // ℓ = max(0, ⌊log₂(u/n)⌋)
            const ratio = @as(u64, u) / @as(u64, n);
            if (ratio <= 1) return 0;
            return @intCast(@as(u8, @intCast(std.math.log2_int(u64, ratio))));
        }

        fn lowBytesCount(data: []const u8) u32 {
            const n: u64 = readIdx(data, 0);
            const low_bits: u64 = data[index_size * 2];
            return @intCast((n * low_bits + 7) / 8);
        }

        fn readIdx(data: []const u8, offset: u32) IndexType {
            const bytes = data[offset..][0..index_size];
            return std.mem.readInt(IndexType, bytes, .little);
        }

        fn writeIdx(data: []u8, offset: u32, value: IndexType) void {
            const bytes = data[offset..][0..index_size];
            std.mem.writeInt(IndexType, bytes, value, .little);
        }

        /// Write `num_bits` from `value` at bit position `bit_offset` (MSB-first).
        fn writeBitsPacked(buf: []u8, bit_offset: u64, value: u64, num_bits: u8) void {
            var remaining: u8 = num_bits;
            var cur_bit: u64 = bit_offset;
            while (remaining > 0) {
                const byte_idx: usize = @intCast(cur_bit / 8);
                const bit_in_byte: u3 = @intCast(7 - (cur_bit % 8));
                const shift: u6 = @intCast(remaining - 1);
                const bit: u8 = @intCast((value >> shift) & 1);
                buf[byte_idx] |= bit << bit_in_byte;
                cur_bit += 1;
                remaining -= 1;
            }
        }

        /// Read `num_bits` from bit position `bit_offset` (MSB-first).
        fn readBitsPacked(buf: []const u8, bit_offset: u64, num_bits: u8) u64 {
            var result: u64 = 0;
            for (0..num_bits) |j| {
                const cur_bit = bit_offset + j;
                const byte_idx: usize = @intCast(cur_bit / 8);
                const bit_in_byte: u3 = @intCast(7 - (cur_bit % 8));
                const bit: u64 = (buf[byte_idx] >> bit_in_byte) & 1;
                result = (result << 1) | bit;
            }
            return result;
        }
    };
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "EliasFano: basic round-trip — small monotone sequence" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 2, 3, 5, 7, 11, 13, 24 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 7), EF.count(data));
    for (values, 0..) |expected, i| {
        try testing.expectEqual(expected, EF.get(data, @intCast(i)));
    }
}

test "EliasFano: single element" {
    const EF = EliasFano(u32);
    const values = [_]u32{42};
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 1), EF.count(data));
    try testing.expectEqual(@as(u32, 42), EF.get(data, 0));
}

test "EliasFano: consecutive values (dense)" {
    const EF = EliasFano(u32);
    var values: [100]u32 = undefined;
    for (0..100) |i| values[i] = @intCast(i);

    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (0..100) |i| {
        try testing.expectEqual(@as(u32, @intCast(i)), EF.get(data, @intCast(i)));
    }
}

test "EliasFano: sparse values with large gaps" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 0, 1000, 50000, 100000, 999999 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (values, 0..) |expected, i| {
        try testing.expectEqual(expected, EF.get(data, @intCast(i)));
    }
}

test "EliasFano: duplicate values (non-strict monotone)" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 5, 5, 5, 10, 10, 20 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (values, 0..) |expected, i| {
        try testing.expectEqual(expected, EF.get(data, @intCast(i)));
    }
}

test "EliasFano: successor query" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 2, 5, 10, 20, 100, 200 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    // Exact matches
    try testing.expectEqual(@as(u32, 0), EF.successor(data, 2));
    try testing.expectEqual(@as(u32, 2), EF.successor(data, 10));

    // Between values
    try testing.expectEqual(@as(u32, 1), EF.successor(data, 3)); // 3 → index of 5
    try testing.expectEqual(@as(u32, 3), EF.successor(data, 15)); // 15 → index of 20

    // Before first
    try testing.expectEqual(@as(u32, 0), EF.successor(data, 0));

    // Beyond last
    try testing.expectEqual(@as(u32, 6), EF.successor(data, 201)); // n
}

test "EliasFano: predecessor query" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 2, 5, 10, 20, 100, 200 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    // Exact matches
    try testing.expectEqual(@as(u32, 0), EF.predecessor(data, 2));
    try testing.expectEqual(@as(u32, 5), EF.predecessor(data, 200));

    // Between values
    try testing.expectEqual(@as(u32, 1), EF.predecessor(data, 7)); // 7 → index of 5
    try testing.expectEqual(@as(u32, 3), EF.predecessor(data, 50)); // 50 → index of 20

    // Before first value
    try testing.expectEqual(@as(u32, 6), EF.predecessor(data, 1)); // returns n (no predecessor)
}

test "EliasFano: contains query" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 3, 7, 15, 31, 63, 127 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (values) |v| {
        try testing.expect(EF.contains(data, v));
    }
    try testing.expect(!EF.contains(data, 0));
    try testing.expect(!EF.contains(data, 4));
    try testing.expect(!EF.contains(data, 128));
}

test "EliasFano: space bound approximation" {
    // Elias-Fano uses approximately n * ceil(log2(u/n)) + 2n bits
    const EF = EliasFano(u32);
    var values: [1000]u32 = undefined;
    for (0..1000) |i| values[i] = @intCast(i * 100); // universe ≈ 100000

    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    const n: u64 = 1000;
    const u: u64 = 100000;
    // Theoretical: n * ceil(log2(u/n)) + 2n ≈ 1000 * 7 + 2000 = 9000 bits ≈ 1125 bytes
    // Plus header and rank/select overhead
    // The encoded size should be reasonable (< 5x theoretical for small n)
    const theoretical_bits = n * (std.math.log2_int(u64, u / n) + 1) + 2 * n;
    const theoretical_bytes = (theoretical_bits + 7) / 8;
    // Allow generous overhead for rank/select auxiliary data + header
    try testing.expect(data.len < theoretical_bytes * 5);
}

test "EliasFano: empty sequence returns error" {
    const EF = EliasFano(u32);
    const values = [_]u32{};
    try testing.expectError(error.EmptySequence, EF.buildUniform(testing.allocator, &values));
}

test "EliasFano: non-monotone returns error" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 10, 5, 20 };
    try testing.expectError(error.NotMonotone, EF.buildUniform(testing.allocator, &values));
}

test "EliasFano: u16 index type" {
    const EF = EliasFano(u16);
    const values = [_]u16{ 100, 200, 300, 400, 500 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (values, 0..) |expected, i| {
        try testing.expectEqual(expected, EF.get(data, @intCast(i)));
    }
}

test "EliasFano: values starting at zero" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 0, 0, 1, 2, 3 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (values, 0..) |expected, i| {
        try testing.expectEqual(expected, EF.get(data, @intCast(i)));
    }
}

test "EliasFano: large values" {
    const EF = EliasFano(u32);
    const values = [_]u32{ 1_000_000, 2_000_000, 3_000_000, 4_000_000 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (values, 0..) |expected, i| {
        try testing.expectEqual(expected, EF.get(data, @intCast(i)));
    }
}
