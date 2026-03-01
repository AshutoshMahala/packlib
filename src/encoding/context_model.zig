//! Context Model — adaptive byte-level context model for entropy estimation.
//!
//! Maintains frequency statistics for byte sequences conditioned on a
//! preceding context of configurable order (0 to 4). Used to estimate
//! conditional entropy and to provide probability distributions for
//! arithmetic / ANS entropy coders.
//!
//! Supports orders 0–4. Higher orders capture more structure but use
//! more memory (256^order entries). Uses Q16.16 fixed-point probabilities
//! for deterministic, portable computation.
//!
//! **Build-time:** allocates frequency tables (uses `ArenaConfig`).
//! **Read-time:** zero allocation for probability queries on built model.
//!
//! ## Memory usage
//!
//! - Order 0: 256 × 4 bytes = 1 KiB (frequencies) + 256 × 4 bytes (cumulative)
//! - Order 1: 256 × 256 × 4 = 256 KiB + similar cumulative
//! - Order 2: 64 KiB context table + sparse
//! - Orders 3-4: hash-based context table
//!
//! ## References
//!
//! - Cleary & Witten (1984), "Data Compression Using Adaptive Coding and Partial String Matching"
//! - Moffat (1990), "Implementing the PPM Data Compression Scheme"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;
const FixedPoint = @import("../math/fixed_point.zig").FixedPoint;

/// Q16.16 fixed-point probability.
const Q16 = FixedPoint(16, 16);

/// Order-0 context model — symbol frequencies without context.
pub const ContextModel0 = struct {
    freqs: [256]u32,
    total: u32,

    pub fn init() ContextModel0 {
        return .{
            .freqs = std.mem.zeroes([256]u32),
            .total = 0,
        };
    }

    /// Train from a byte sequence.
    pub fn train(input: []const u8) ContextModel0 {
        var self = init();
        for (input) |c| {
            self.freqs[c] += 1;
            self.total += 1;
        }
        return self;
    }

    /// Probability of symbol `c` as Q16.16.
    /// Returns 0 if total is 0.
    pub fn prob(self: *const ContextModel0, c: u8) Q16 {
        if (self.total == 0) return Q16.fromInt(0);
        // P(c) = freq[c] / total, scaled to Q16.16
        const num: i64 = @as(i64, self.freqs[c]) << 16;
        const den: i64 = @intCast(self.total);
        return .{ .raw = @intCast(@divTrunc(num, den)) };
    }

    /// Number of distinct symbols observed.
    pub fn alphabetSize(self: *const ContextModel0) u16 {
        var count: u16 = 0;
        for (self.freqs) |f| {
            if (f > 0) count += 1;
        }
        return count;
    }

    /// Get cumulative frequency up to (but not including) symbol `c`.
    /// Used by ANS/arithmetic coders.
    pub fn cumulativeFreq(self: *const ContextModel0, c: u8) u32 {
        var cum: u32 = 0;
        for (0..@as(usize, c)) |i| {
            cum += self.freqs[i];
        }
        return cum;
    }
};

/// Order-1 context model — symbol frequencies conditioned on previous byte.
pub const ContextModel1 = struct {
    /// freqs[context][symbol] = count of symbol after context byte
    freqs: [256][256]u32,
    /// totals[context] = total count for this context
    totals: [256]u32,

    pub fn init() ContextModel1 {
        return .{
            .freqs = std.mem.zeroes([256][256]u32),
            .totals = std.mem.zeroes([256]u32),
        };
    }

    /// Train from a byte sequence.
    pub fn train(input: []const u8) ContextModel1 {
        var self = init();
        for (1..input.len) |i| {
            const ctx = input[i - 1];
            const sym = input[i];
            self.freqs[ctx][sym] += 1;
            self.totals[ctx] += 1;
        }
        return self;
    }

    /// Conditional probability P(symbol | context) as Q16.16.
    pub fn prob(self: *const ContextModel1, context: u8, symbol: u8) Q16 {
        if (self.totals[context] == 0) return Q16.fromInt(0);
        const num: i64 = @as(i64, self.freqs[context][symbol]) << 16;
        const den: i64 = @intCast(self.totals[context]);
        return .{ .raw = @intCast(@divTrunc(num, den)) };
    }

    /// Cumulative frequency of `symbol` within `context`.
    pub fn cumulativeFreq(self: *const ContextModel1, context: u8, symbol: u8) u32 {
        var cum: u32 = 0;
        for (0..@as(usize, symbol)) |i| {
            cum += self.freqs[context][i];
        }
        return cum;
    }

    /// Total count for a given context.
    pub fn contextTotal(self: *const ContextModel1, context: u8) u32 {
        return self.totals[context];
    }
};

/// Heap-allocated higher-order context model (orders 2–4).
/// Uses a hash map for sparse context storage.
pub fn ContextModelN(comptime order: u8) type {
    comptime {
        if (order < 2 or order > 4) @compileError("ContextModelN supports orders 2–4");
    }

    const ContextKey = switch (order) {
        2 => u16,
        3 => u24,
        4 => u32,
        else => unreachable,
    };

    return struct {
        const Self = @This();

        /// Per-context frequency table.
        const ContextEntry = struct {
            freqs: [256]u32,
            total: u32,
        };

        map: std.AutoHashMapUnmanaged(ContextKey, ContextEntry),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .map = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.map.deinit(self.allocator);
        }

        /// Train from a byte sequence using ArenaConfig.
        pub fn trainArena(arena: ArenaConfig, input: []const u8) !Self {
            return trainImpl(arena.temp, input);
        }

        /// Train from a byte sequence.
        pub fn trainAlloc(allocator: std.mem.Allocator, input: []const u8) !Self {
            return trainImpl(allocator, input);
        }

        fn trainImpl(allocator: std.mem.Allocator, input: []const u8) !Self {
            var self = init(allocator);
            errdefer self.deinit();

            if (input.len <= order) return self;

            for (order..input.len) |i| {
                const key = extractKey(input, i);
                const sym = input[i];

                const result = try self.map.getOrPut(self.allocator, key);
                if (!result.found_existing) {
                    result.value_ptr.* = .{
                        .freqs = std.mem.zeroes([256]u32),
                        .total = 0,
                    };
                }
                result.value_ptr.freqs[sym] += 1;
                result.value_ptr.total += 1;
            }

            return self;
        }

        /// Conditional probability P(symbol | context) as Q16.16.
        pub fn prob(self: *const Self, context: ContextKey, symbol: u8) Q16 {
            const entry = self.map.get(context) orelse return Q16.fromInt(0);
            if (entry.total == 0) return Q16.fromInt(0);
            const num: i64 = @as(i64, entry.freqs[symbol]) << 16;
            const den: i64 = @intCast(entry.total);
            return .{ .raw = @intCast(@divTrunc(num, den)) };
        }

        /// Number of unique contexts observed.
        pub fn numContexts(self: *const Self) u32 {
            return self.map.count();
        }

        fn extractKey(input: []const u8, pos: usize) ContextKey {
            var key: ContextKey = 0;
            inline for (0..order) |j| {
                key = (key << 8) | @as(ContextKey, input[pos - order + j]);
            }
            return key;
        }
    };
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "ContextModel0: basic training" {
    const model = ContextModel0.train("aabbc");

    try testing.expectEqual(@as(u32, 2), model.freqs['a']);
    try testing.expectEqual(@as(u32, 2), model.freqs['b']);
    try testing.expectEqual(@as(u32, 1), model.freqs['c']);
    try testing.expectEqual(@as(u32, 5), model.total);
}

test "ContextModel0: probability" {
    const model = ContextModel0.train("aabb");

    const p_a = model.prob('a');
    const p_b = model.prob('b');

    // Both should be 0.5 = 0x8000 in Q16.16
    try testing.expectEqual(@as(i32, 0x8000), p_a.raw);
    try testing.expectEqual(@as(i32, 0x8000), p_b.raw);

    // Zero frequency
    const p_z = model.prob('z');
    try testing.expectEqual(@as(i32, 0), p_z.raw);
}

test "ContextModel0: alphabet size" {
    const model = ContextModel0.train("hello");
    try testing.expectEqual(@as(u16, 4), model.alphabetSize()); // h, e, l, o
}

test "ContextModel0: cumulative frequency" {
    const model = ContextModel0.train("aabbc");

    // Cumulative before 'a' (0x61) should be sum of freqs[0..0x61)
    // Only 'a','b','c' have nonzero freqs, all above 0x61
    // Actually: 'a'=0x61, 'b'=0x62, 'c'=0x63
    try testing.expectEqual(@as(u32, 0), model.cumulativeFreq('a'));
    try testing.expectEqual(@as(u32, 2), model.cumulativeFreq('b')); // 2 'a's before 'b'
    try testing.expectEqual(@as(u32, 4), model.cumulativeFreq('c')); // 2 'a's + 2 'b's
}

test "ContextModel1: basic training" {
    const model = ContextModel1.train("abab");

    // After 'a', we see 'b' twice
    try testing.expectEqual(@as(u32, 2), model.freqs['a']['b']);
    // After 'b', we see 'a' once
    try testing.expectEqual(@as(u32, 1), model.freqs['b']['a']);
}

test "ContextModel1: probability" {
    const model = ContextModel1.train("ababab");

    // After 'a': always 'b' → P(b|a) = 1.0
    const p_b_given_a = model.prob('a', 'b');
    // Q16.16: 1.0 = 0x10000, but due to integer division it may be 0xFFFF
    try testing.expect(p_b_given_a.raw >= 0xFFF0); // close to 1.0

    // After 'b': always 'a' → P(a|b) ≈ 1.0
    const p_a_given_b = model.prob('b', 'a');
    try testing.expect(p_a_given_b.raw >= 0xFFF0);

    // P(c|a) = 0 for c != 'b'
    try testing.expectEqual(@as(i32, 0), model.prob('a', 'c').raw);
}

test "ContextModelN(2): basic training" {
    var model = try ContextModelN(2).trainAlloc(testing.allocator, "abcabc");
    defer model.deinit();

    // Context "ab" (0x6162) → symbol 'c'
    const key_ab: u16 = (@as(u16, 'a') << 8) | 'b';
    const p = model.prob(key_ab, 'c');
    try testing.expect(p.raw > 0);
}

test "ContextModelN(2): num contexts" {
    var model = try ContextModelN(2).trainAlloc(testing.allocator, "abcabc");
    defer model.deinit();

    // Contexts at positions 2,3,4,5: "ab","bc","ca","ab","bc"
    // Unique: "ab", "bc", "ca" = 3
    try testing.expectEqual(@as(u32, 3), model.numContexts());
}

test "ContextModelN(3): training" {
    var model = try ContextModelN(3).trainAlloc(testing.allocator, "abcdabcd");
    defer model.deinit();

    try testing.expect(model.numContexts() > 0);
}

test "ContextModel0: empty input" {
    const model = ContextModel0.train("");
    try testing.expectEqual(@as(u32, 0), model.total);
    try testing.expectEqual(@as(u16, 0), model.alphabetSize());
}

test "ContextModel1: single byte input" {
    const model = ContextModel1.train("a");
    // No bigrams → all totals are 0
    for (model.totals) |t| {
        try testing.expectEqual(@as(u32, 0), t);
    }
}

test "ContextModel0: probabilities sum approximately to 1" {
    const model = ContextModel0.train("the quick brown fox jumps over the lazy dog");

    var sum: i64 = 0;
    for (0..256) |i| {
        sum += model.prob(@intCast(i)).raw;
    }
    // Should be close to 1.0 (0x10000 in Q16.16)
    // Allow small rounding error
    try testing.expect(sum >= 0xFF00);
    try testing.expect(sum <= 0x10100);
}
