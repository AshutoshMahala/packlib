//! Shannon entropy analysis — offline diagnostic tool.
//!
//! Computes 0th through Nth order entropy of byte sequences.
//! Used to evaluate theoretical compression limits and inform algorithm selection.
//! Not part of any runtime encode/decode pipeline.

const std = @import("std");
const math = std.math;

pub const EntropyReport = struct {
    h0: f64, // 0th-order entropy (bits/symbol)
    h1: f64, // 1st-order entropy (bits/symbol)
    total_symbols: u64,
    unique_symbols: u16,
    theoretical_min_bytes: u64, // ceil(h0 * total_symbols / 8)
};

pub const Entropy = struct {
    /// Compute entropy report for a byte sequence.
    pub fn analyze(allocator: std.mem.Allocator, data: []const u8) !EntropyReport {
        if (data.len == 0) {
            return EntropyReport{
                .h0 = 0,
                .h1 = 0,
                .total_symbols = 0,
                .unique_symbols = 0,
                .theoretical_min_bytes = 0,
            };
        }

        const h0 = computeH0(data);
        const h1 = try computeH1(allocator, data);

        var unique: u16 = 0;
        var freq: [256]u64 = [_]u64{0} ** 256;
        for (data) |b| freq[b] += 1;
        for (freq) |f| {
            if (f > 0) unique += 1;
        }

        const total: u64 = data.len;
        const total_bits = h0 * @as(f64, @floatFromInt(total));
        const min_bytes: u64 = if (total_bits == 0) 0 else @intFromFloat(@ceil(total_bits / 8.0));

        return EntropyReport{
            .h0 = h0,
            .h1 = h1,
            .total_symbols = total,
            .unique_symbols = unique,
            .theoretical_min_bytes = min_bytes,
        };
    }

    /// Compute Nth-order entropy.
    pub fn entropyN(allocator: std.mem.Allocator, data: []const u8, order: u8) !f64 {
        if (data.len == 0) return 0;
        if (order == 0) return computeH0(data);
        if (order == 1) return computeH1(allocator, data);

        // General Nth-order: H_n = H(n+1-gram) - H(n-gram)
        // where H(k-gram) is the entropy of k-length subsequences.
        const h_n1 = try ngramEntropy(allocator, data, @as(u32, order) + 1);
        const h_n = try ngramEntropy(allocator, data, order);
        return h_n1 - h_n;
    }

    /// Compute per-position entropy across a set of strings.
    /// Returns entropy for each character position.
    /// Caller owns the returned slice.
    pub fn perPositionEntropy(allocator: std.mem.Allocator, strings: []const []const u8) ![]f64 {
        if (strings.len == 0) return try allocator.alloc(f64, 0);

        // Find max string length
        var max_len: usize = 0;
        for (strings) |s| {
            if (s.len > max_len) max_len = s.len;
        }

        const result = try allocator.alloc(f64, max_len);

        for (0..max_len) |pos| {
            // Collect all characters at this position
            var freq: [256]u64 = [_]u64{0} ** 256;
            var count: u64 = 0;
            for (strings) |s| {
                if (pos < s.len) {
                    freq[s[pos]] += 1;
                    count += 1;
                }
            }

            if (count == 0) {
                result[pos] = 0;
                continue;
            }

            // Compute entropy over this position's distribution
            var h: f64 = 0;
            const n: f64 = @floatFromInt(count);
            for (freq) |f| {
                if (f > 0) {
                    const p: f64 = @as(f64, @floatFromInt(f)) / n;
                    h -= p * @log2(p);
                }
            }
            result[pos] = h;
        }

        return result;
    }

    // ── Internal helpers ──

    /// 0th-order entropy: H₀ = -Σ p(x) log₂ p(x)
    fn computeH0(data: []const u8) f64 {
        var freq: [256]u64 = [_]u64{0} ** 256;
        for (data) |b| freq[b] += 1;

        const n: f64 = @floatFromInt(data.len);
        var h: f64 = 0;
        for (freq) |f| {
            if (f > 0) {
                const p: f64 = @as(f64, @floatFromInt(f)) / n;
                h -= p * @log2(p);
            }
        }
        return h;
    }

    /// 1st-order entropy: H₁ = Σ_c P(c) × H(next | c)
    /// Uses allocator for bigram count table.
    fn computeH1(allocator: std.mem.Allocator, data: []const u8) !f64 {
        if (data.len < 2) return 0;

        // Bigram counts: bigrams[prev][next]
        const bigrams = try allocator.alloc([256]u64, 256);
        defer allocator.free(bigrams);
        for (bigrams) |*row| {
            row.* = [_]u64{0} ** 256;
        }

        // Context counts
        var context_count: [256]u64 = [_]u64{0} ** 256;

        for (0..data.len - 1) |i| {
            const prev = data[i];
            const next = data[i + 1];
            bigrams[prev][next] += 1;
            context_count[prev] += 1;
        }

        // H₁ = Σ_c (count_c / total) × H(next | c)
        const total: f64 = @floatFromInt(data.len - 1);
        var h1: f64 = 0;

        for (0..256) |c| {
            const cc = context_count[c];
            if (cc == 0) continue;

            const cc_f: f64 = @floatFromInt(cc);
            var h_cond: f64 = 0;

            for (0..256) |next| {
                const bc = bigrams[c][next];
                if (bc > 0) {
                    const p: f64 = @as(f64, @floatFromInt(bc)) / cc_f;
                    h_cond -= p * @log2(p);
                }
            }

            h1 += (cc_f / total) * h_cond;
        }

        return h1;
    }

    /// Compute the per-symbol entropy of k-grams: H(k-gram) / k isn't quite
    /// right for conditional entropy. Instead, H(k-gram) is the joint entropy
    /// of all k-length subsequences, useful for H_n = H(n+1-gram) - H(n-gram).
    fn ngramEntropy(allocator: std.mem.Allocator, data: []const u8, k: u32) !f64 {
        if (data.len < k) return 0;

        // Use a hash map to count k-gram occurrences
        const Map = std.StringHashMap(u64);
        var map = Map.init(allocator);
        defer {
            // Free all keys (we duped them)
            var key_it = map.keyIterator();
            while (key_it.next()) |key_ptr| {
                allocator.free(key_ptr.*);
            }
            map.deinit();
        }

        const ngram_count: u64 = data.len - k + 1;

        for (0..@intCast(ngram_count)) |i| {
            const gram = data[i .. i + k];
            // We need to dupe the key since the data slice may alias
            const entry = try map.getOrPut(gram);
            if (entry.found_existing) {
                entry.value_ptr.* += 1;
            } else {
                entry.key_ptr.* = try allocator.dupe(u8, gram);
                entry.value_ptr.* = 1;
            }
        }

        const total: f64 = @floatFromInt(ngram_count);
        var h: f64 = 0;
        var val_it = map.valueIterator();
        while (val_it.next()) |count_ptr| {
            const p: f64 = @as(f64, @floatFromInt(count_ptr.*)) / total;
            h -= p * @log2(p);
        }

        return h;
    }
};

// ── Tests ────────────────────────────────────────────────────
const testing = std.testing;

test "Entropy: uniform distribution" {
    // "abcabc" — 3 unique symbols, uniform distribution
    // H₀ = log₂(3) ≈ 1.58496
    const data = "abcabc";
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(@abs(report.h0 - @log2(@as(f64, 3.0))) < 0.001);
    try testing.expectEqual(@as(u16, 3), report.unique_symbols);
    try testing.expectEqual(@as(u64, 6), report.total_symbols);
}

test "Entropy: single character" {
    // "aaaa" — H₀ = 0
    const data = "aaaa";
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(@abs(report.h0) < 0.001);
    try testing.expectEqual(@as(u16, 1), report.unique_symbols);
}

test "Entropy: empty input" {
    const data = "";
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(report.h0 == 0);
    try testing.expectEqual(@as(u64, 0), report.total_symbols);
}

test "Entropy: H1 < H0 for natural text" {
    // Repeated pattern creates strong bigram dependencies
    const data = "the the the the the the the the ";
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(report.h1 < report.h0);
}

test "Entropy: H1 for single character is 0" {
    const data = "aaaa";
    const h1 = try Entropy.entropyN(testing.allocator, data, 1);
    try testing.expect(@abs(h1) < 0.001);
}

test "Entropy: higher order decreasing" {
    // For natural-language-like data, H₂ < H₁ < H₀
    const data = "abracadabra abracadabra abracadabra ";
    const h0 = try Entropy.entropyN(testing.allocator, data, 0);
    const h1 = try Entropy.entropyN(testing.allocator, data, 1);
    const h2 = try Entropy.entropyN(testing.allocator, data, 2);
    try testing.expect(h0 >= h1);
    try testing.expect(h1 >= h2);
}

test "Entropy: theoretical_min_bytes" {
    // "ab" repeated → H₀ = 1.0 bits/symbol, 8 symbols → 1 byte min
    const data = "abababab";
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(@abs(report.h0 - 1.0) < 0.001);
    try testing.expectEqual(@as(u64, 1), report.theoretical_min_bytes);
}

test "Entropy: perPositionEntropy" {
    const strings = [_][]const u8{ "abc", "abd", "aec" };
    const result = try Entropy.perPositionEntropy(testing.allocator, &strings);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 3), result.len);
    // Position 0: all 'a' → entropy 0
    try testing.expect(@abs(result[0]) < 0.001);
    // Position 1: 'b','b','e' → 2 unique, not uniform
    try testing.expect(result[1] > 0);
    // Position 2: 'c','d','c' → 2 unique
    try testing.expect(result[2] > 0);
}
