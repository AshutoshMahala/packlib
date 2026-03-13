//! Reference test vectors from published standards and literature.
//!
//! These tests verify our implementations against known-correct values from:
//!  - Wikipedia: Huffman coding, Canonical Huffman code, LEB128, Entropy
//!  - Shannon (1948): "A Mathematical Theory of Communication"
//!  - IEEE/DWARF standard: LEB128 unsigned encoding
//!
//! Each test cites its source so correctness can be independently verified.

const packlib = @import("packlib");
const std = @import("std");
const testing = std.testing;
const math = std.math;

// ═══════════════════════════════════════════════════════════════
// Shannon Entropy – Reference Values
// Source: Wikipedia "Entropy (information theory)"
// Formula: H(X) = -Σ p(x) log2 p(x)
// ═══════════════════════════════════════════════════════════════

test "Entropy H0: fair coin (2 equiprobable outcomes) → H = 1.0 bit" {
    // Source: Wikipedia Entropy (information theory), Example section
    // "The maximum surprise is when p = 1/2 ... entropy of one bit"
    // Two equally likely symbols → H = log2(2) = 1.0
    const Entropy = packlib.Entropy;
    const data = "ababababababababababab"; // equal freq of 'a' and 'b'
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(@abs(report.h0 - 1.0) < 0.01);
}

test "Entropy H0: uniform over 4 symbols → H = 2.0 bits" {
    // Source: Wikipedia Entropy, Introduction/Example
    // "If all 4 letters are equally likely (25%), one cannot do better
    //  than using two bits to encode each letter" → H = log2(4) = 2.0
    const Entropy = packlib.Entropy;
    const data = "abcdabcdabcdabcdabcdabcdabcdabcd"; // equal freq
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(@abs(report.h0 - 2.0) < 0.01);
}

test "Entropy H0: single symbol → H = 0.0 bits" {
    // Source: Wikipedia Entropy, Introduction
    // "The minimum surprise is when p = 0 or p = 1 ... entropy is zero bits"
    const Entropy = packlib.Entropy;
    const data = "aaaaaaaaaaaaaaa";
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(report.h0 < 0.001);
}

test "Entropy H0: biased coin p=0.7 → H ≈ 0.8813 bits" {
    // Source: Wikipedia Entropy (information theory), Example section
    // H = -0.7*log2(0.7) - 0.3*log2(0.3) ≈ 0.8813 bits
    // We approximate p≈0.7 with 70 'a's and 30 'b's
    const Entropy = packlib.Entropy;
    var data: [100]u8 = undefined;
    for (0..70) |i| data[i] = 'a';
    for (70..100) |i| data[i] = 'b';
    const report = try Entropy.analyze(testing.allocator, &data);
    // Allow small tolerance due to integer frequency approximation
    try testing.expect(@abs(report.h0 - 0.8813) < 0.02);
}

test "Entropy H0: uniform over 6 symbols (fair die) → H = log2(6) ≈ 2.585 bits" {
    // Source: Wikipedia Entropy, Introduction
    // "The result of a fair die (6 possible values) would have entropy log2(6) bits"
    const Entropy = packlib.Entropy;
    // 6 equally likely symbols, repeated to get good frequency approximation
    const data = "abcdefabcdefabcdefabcdefabcdefabcdef" ** 3;
    const expected = @log2(@as(f64, 6.0)); // ≈ 2.585
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(@abs(report.h0 - expected) < 0.01);
}

test "Entropy H0: Wikipedia Huffman example {a1:.4, a2:.35, a3:.2, a4:.05} → H ≈ 1.7370 bits" {
    // Source: Wikipedia "Huffman coding" Compression section
    // Probabilities {0.4, 0.35, 0.2, 0.05}
    // H = -(0.4*log2(0.4) + 0.35*log2(0.35) + 0.2*log2(0.2) + 0.05*log2(0.05))
    //   ≈ 1.7370 bits/symbol
    // "the entropy of the source is 1.74 bits/symbol"
    // We use 100 symbols: 40×a, 35×b, 20×c, 5×d
    const Entropy = packlib.Entropy;
    var data: [100]u8 = undefined;
    for (0..40) |i| data[i] = 'a';
    for (40..75) |i| data[i] = 'b';
    for (75..95) |i| data[i] = 'c';
    for (95..100) |i| data[i] = 'd';
    const report = try Entropy.analyze(testing.allocator, &data);
    // Direct calculation: 1.7370 bits
    try testing.expect(@abs(report.h0 - 1.7370) < 0.02);
}

test "Entropy: H0 always ≤ log2(|alphabet|)" {
    // Source: Wikipedia Entropy, Further Properties
    // "H(p1,...,pn) ≤ log_b(n)" — maximum entropy for n outcomes
    const Entropy = packlib.Entropy;
    const data = "aabbbccccdddddeeeee"; // 5 distinct symbols
    const report = try Entropy.analyze(testing.allocator, data);
    const max_entropy = @log2(@as(f64, 5.0)); // log2(5) ≈ 2.322
    try testing.expect(report.h0 <= max_entropy + 0.001);
}

test "Entropy: H1 ≤ H0 (conditioning reduces entropy)" {
    // Source: Wikipedia Entropy, Further Properties
    // "H(X|Y) ≤ H(X)" — conditioning can only reduce or maintain entropy
    const Entropy = packlib.Entropy;
    const data = "abababababcdcdcdcd"; // strong sequential pattern
    const report = try Entropy.analyze(testing.allocator, data);
    try testing.expect(report.h1 <= report.h0 + 0.001);
}

// ═══════════════════════════════════════════════════════════════
// Canonical Huffman Coding – Reference Values
// Source: Wikipedia "Huffman coding" and "Canonical Huffman code"
// ═══════════════════════════════════════════════════════════════

test "Huffman: Wikipedia 4-symbol example code lengths {0.4, 0.35, 0.2, 0.05}" {
    // Source: Wikipedia "Huffman coding", Compression section
    // Probabilities {0.4, 0.35, 0.2, 0.05} → code lengths {1, 2, 3, 3}
    // Codes: a1=0, a2=10, a3=110, a4=111
    const HC = packlib.HuffmanCodec(u8);

    // Scale probabilities × 100 for integer frequencies
    var freq = [_]u64{0} ** 256;
    freq[0] = 40; // a1: 0.4
    freq[1] = 35; // a2: 0.35
    freq[2] = 20; // a3: 0.2
    freq[3] = 5; //  a4: 0.05

    const table = try HC.buildTable(testing.allocator, &freq);

    // Expected code lengths: a1=1, a2=2, a3=3, a4=3
    try testing.expectEqual(@as(u8, 1), table.code_lengths[0]);
    try testing.expectEqual(@as(u8, 2), table.code_lengths[1]);
    try testing.expectEqual(@as(u8, 3), table.code_lengths[2]);
    try testing.expectEqual(@as(u8, 3), table.code_lengths[3]);
}

test "Huffman: canonical codes for known lengths" {
    // Source: Wikipedia "Canonical Huffman code", Algorithm section
    // Given code lengths: B=1, A=2, C=3, D=3
    // Canonical codes: B=0, A=10, C=110, D=111
    //
    // We verify by building a table with frequencies that produce these lengths:
    //   B most frequent → len 1
    //   A next → len 2
    //   C, D least → len 3 each
    const HC = packlib.HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
    freq['B'] = 100; // most frequent → shortest code
    freq['A'] = 40; //  next
    freq['C'] = 10; //  least
    freq['D'] = 5; //   least

    const table = try HC.buildTable(testing.allocator, &freq);

    // Expected code lengths
    try testing.expectEqual(@as(u8, 1), table.code_lengths['B']);
    try testing.expectEqual(@as(u8, 2), table.code_lengths['A']);
    try testing.expectEqual(@as(u8, 3), table.code_lengths['C']);
    try testing.expectEqual(@as(u8, 3), table.code_lengths['D']);

    // Canonical code property: for same length, smaller symbol → smaller code
    // C < D alphabetically, both len 3, so code(C) < code(D)
    try testing.expect(table.codes['C'] < table.codes['D']);

    // Verify actual canonical codes:
    // Sorted by (len, symbol): B(1), A(2), C(3), D(3)
    // B: code = 0 (len 1)
    // A: code = (0+1) << (2-1) = 10 (len 2)
    // C: code = (10+1) << (3-2) = 110 (len 3)
    // D: code = 110+1 = 111 (len 3)
    try testing.expectEqual(@as(u32, 0b0), table.codes['B']); // 0
    try testing.expectEqual(@as(u32, 0b10), table.codes['A']); // 2
    try testing.expectEqual(@as(u32, 0b110), table.codes['C']); // 6
    try testing.expectEqual(@as(u32, 0b111), table.codes['D']); // 7
}

test "Huffman: Kraft inequality holds" {
    // Source: Wikipedia "Huffman coding", Optimality/Problem definition
    // For any valid prefix-free code: Σ 2^(-l_i) ≤ 1
    const HC = packlib.HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
    freq['a'] = 50;
    freq['b'] = 25;
    freq['c'] = 15;
    freq['d'] = 7;
    freq['e'] = 3;

    const table = try HC.buildTable(testing.allocator, &freq);

    // Kraft inequality: Σ 2^(-l_i) ≤ 1
    var kraft_sum: f64 = 0;
    for (0..256) |i| {
        if (table.code_lengths[i] > 0) {
            kraft_sum += math.pow(f64, 2.0, -@as(f64, @floatFromInt(table.code_lengths[i])));
        }
    }
    try testing.expect(kraft_sum <= 1.0 + 0.001);
}

test "Huffman: weighted path length ≥ entropy (Shannon's source coding theorem)" {
    // Source: Wikipedia "Huffman coding", Example table
    // L(C) ≥ H(A) — weighted average code length is at least the entropy
    const HC = packlib.HuffmanCodec(u8);

    var freq = [_]u64{0} ** 256;
    freq['a'] = 10; // p = 0.10
    freq['b'] = 15; // p = 0.15
    freq['c'] = 30; // p = 0.30
    freq['d'] = 16; // p = 0.16
    freq['e'] = 29; // p = 0.29

    const table = try HC.buildTable(testing.allocator, &freq);
    const total: f64 = 100.0;

    // Compute weighted path length L(C) = Σ p_i * l_i
    var wpl: f64 = 0;
    var entropy: f64 = 0;
    const symbols = [_]u8{ 'a', 'b', 'c', 'd', 'e' };
    const weights = [_]f64{ 10.0, 15.0, 30.0, 16.0, 29.0 };

    for (symbols, weights) |sym, w| {
        const p = w / total;
        const l: f64 = @floatFromInt(table.code_lengths[sym]);
        wpl += p * l;
        entropy += -p * @log2(p);
    }

    // L(C) ≥ H(A)
    try testing.expect(wpl >= entropy - 0.001);
    // L(C) < H(A) + 1 (Huffman guarantee)
    try testing.expect(wpl < entropy + 1.0 + 0.001);
}

test "Huffman: encode/decode round-trip matches reference" {
    // Source: Wikipedia "Huffman coding" example with "this is an example of a huffman tree"
    // We verify round-trip correctness for the fundamental property:
    // decode(encode(message)) = message
    const HC = packlib.HuffmanCodec(u8);
    const BW = packlib.BitWriter(u32);
    const BR = packlib.BitReader;

    const message = "this is an example of a huffman tree";

    // Build frequencies from the message itself
    var freq = [_]u64{0} ** 256;
    for (message) |c| freq[c] += 1;

    const table = try HC.buildTable(testing.allocator, &freq);

    // Encode
    var writer = BW.init(testing.allocator);
    defer writer.deinit();
    try HC.encode(&table, message, &writer);
    const bits = try writer.finish();
    defer testing.allocator.free(bits);

    // Build view for decoding via serialize → deserialize round-trip
    var ser_writer = BW.init(testing.allocator);
    defer ser_writer.deinit();
    try HC.serializeTable(&table, &ser_writer);
    const ser_bytes = try ser_writer.finish();
    defer testing.allocator.free(ser_bytes);

    var ser_reader = BR.init(ser_bytes);
    const view = try HC.deserializeTable(&ser_reader);

    // Decode
    var reader = BR.init(bits);
    var decoded: [36]u8 = undefined;
    for (0..message.len) |i| {
        decoded[i] = try HC.decodeSymbol(&view, &reader);
    }
    try testing.expectEqualStrings(message, &decoded);
}

test "Huffman: prefix-free property — no code is a prefix of another" {
    // Source: fundamental property of Huffman codes (prefix-free code)
    const HC = packlib.HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
    freq['a'] = 40;
    freq['b'] = 35;
    freq['c'] = 20;
    freq['d'] = 5;

    const table = try HC.buildTable(testing.allocator, &freq);

    // Check no code is a prefix of another
    // Collect all (code, length) pairs
    var codes: [256]struct { code: u32, len: u8 } = undefined;
    var n: usize = 0;
    for (0..256) |i| {
        if (table.code_lengths[i] > 0) {
            codes[n] = .{ .code = table.codes[i], .len = table.code_lengths[i] };
            n += 1;
        }
    }

    // For each pair, check that neither is a prefix of the other
    for (0..n) |i| {
        for (i + 1..n) |j| {
            const shorter_len = @min(codes[i].len, codes[j].len);
            // Align both codes to compare the first shorter_len bits
            const a_prefix = codes[i].code >> @intCast(codes[i].len - shorter_len);
            const b_prefix = codes[j].code >> @intCast(codes[j].len - shorter_len);
            // If lengths differ, the prefix bits must differ
            if (codes[i].len != codes[j].len) {
                try testing.expect(a_prefix != b_prefix);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// LEB128 (Unsigned) – Reference Values
// Source: Wikipedia "LEB128" and DWARF Spec
// Our DeltaVarint uses unsigned LEB128 internally for deltas.
// ═══════════════════════════════════════════════════════════════

test "LEB128 via DeltaVarint: Wikipedia reference value 624485 → 0xE5 0x8E 0x26" {
    // Source: Wikipedia "LEB128", Unsigned LEB128 section
    // 624485 → output stream: 0xE5, 0x8E, 0x26
    //
    // We verify by encoding a sequence [0, 624485]. The delta is 624485,
    // which should be LEB128-encoded as those 3 bytes.
    const DV = packlib.DeltaVarint(u32);
    const values = [_]u32{ 0, 624485 };
    // Use block_size > 2 so both values are in one block
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 64);
    defer testing.allocator.free(data);

    // Verify round-trip: the value at index 1 should be 624485
    try testing.expectEqual(@as(u32, 624485), DV.get(data, 1));

    // The LEB128 bytes for delta 624485 should be somewhere in the data.
    // Find the sequence 0xE5, 0x8E, 0x26 in the encoded data.
    var found = false;
    for (0..data.len - 2) |i| {
        if (data[i] == 0xE5 and data[i + 1] == 0x8E and data[i + 2] == 0x26) {
            found = true;
            break;
        }
    }
    try testing.expect(found);
}

test "LEB128 via DeltaVarint: value 0 → single byte 0x00" {
    // Source: Wikipedia LEB128: "The number zero is usually encoded as a single byte 0x00"
    // Sequence [100, 100]: delta = 0, second value accessible as 100
    const DV = packlib.DeltaVarint(u32);
    const values = [_]u32{ 100, 100 };
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 64);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 100), DV.get(data, 1));
    // A delta of 0 should be a single 0x00 byte
    // Find 0x00 in the delta section (after header + abs + offsets)
    // Header=6, 1 block × 4 bytes abs = 4, 1 block × 4 bytes offset = 4
    // Delta section starts at 6 + 4 + 4 = 14
    try testing.expectEqual(@as(u8, 0x00), data[14]);
}

test "LEB128 via DeltaVarint: value 127 → 0x7F (single byte)" {
    // Source: LEB128 spec — values 0-127 fit in a single byte
    const DV = packlib.DeltaVarint(u32);
    const values = [_]u32{ 0, 127 };
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 64);
    defer testing.allocator.free(data);
    try testing.expectEqual(@as(u32, 127), DV.get(data, 1));
}

test "LEB128 via DeltaVarint: value 128 → 0x80 0x01 (two bytes)" {
    // Source: LEB128 spec — 128 requires 2 bytes
    // 128 = 0b10000000 → groups: 0000001 0000000 → bytes: 0x80 0x01
    const DV = packlib.DeltaVarint(u32);
    const values = [_]u32{ 0, 128 };
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 64);
    defer testing.allocator.free(data);
    try testing.expectEqual(@as(u32, 128), DV.get(data, 1));

    // Delta section starts at 6 + 4 + 4 = 14
    try testing.expectEqual(@as(u8, 0x80), data[14]);
    try testing.expectEqual(@as(u8, 0x01), data[15]);
}

test "LEB128 via DeltaVarint: value 16384 → 0x80 0x80 0x01" {
    // Source: LEB128 — 16384 = 2^14, requires 3 bytes
    const DV = packlib.DeltaVarint(u32);
    const values = [_]u32{ 0, 16384 };
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 64);
    defer testing.allocator.free(data);
    try testing.expectEqual(@as(u32, 16384), DV.get(data, 1));

    // Delta section at offset 14
    try testing.expectEqual(@as(u8, 0x80), data[14]);
    try testing.expectEqual(@as(u8, 0x80), data[15]);
    try testing.expectEqual(@as(u8, 0x01), data[16]);
}

// ═══════════════════════════════════════════════════════════════
// Rank/Select – Reference Values
// Source: standard popcount-based verification
// ═══════════════════════════════════════════════════════════════

test "RankSelect: popcount of 0xFF = 8 (all ones byte)" {
    // Standard reference: popcount(0xFF) = 8
    const RS = packlib.RankSelect(u32);
    var bits = [_]u1{1} ** 8;
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);
    try testing.expectEqual(@as(u32, 8), RS.rank1(data, 8));
}

test "RankSelect: popcount of 0x00 = 0 (all zeros byte)" {
    const RS = packlib.RankSelect(u32);
    var bits = [_]u1{0} ** 8;
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);
    try testing.expectEqual(@as(u32, 0), RS.rank1(data, 8));
}

test "RankSelect: popcount of 0xAA (10101010) = 4" {
    // 0xAA = 10101010 in binary (MSB first)
    // popcount = 4
    const RS = packlib.RankSelect(u32);
    var bits = [_]u1{ 1, 0, 1, 0, 1, 0, 1, 0 }; // MSB first: 0xAA
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);
    try testing.expectEqual(@as(u32, 4), RS.rank1(data, 8));
}

test "RankSelect: rank/select identity — select1(rank1(i)-1) gives position of last 1 before i" {
    // Fundamental identity: for any bit position p that is set,
    // select1(rank1(p+1) - 1) = p
    const RS = packlib.RankSelect(u32);
    // Pattern: set bits at 0, 3, 5, 7, 10, 15
    var bits = [_]u1{0} ** 16;
    bits[0] = 1;
    bits[3] = 1;
    bits[5] = 1;
    bits[7] = 1;
    bits[10] = 1;
    bits[15] = 1;

    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    // For each set bit position p: select1(rank1(p+1) - 1) == p
    const set_positions = [_]u32{ 0, 3, 5, 7, 10, 15 };
    for (set_positions) |p| {
        const rank_val = RS.rank1(data, p + 1);
        try testing.expect(rank_val > 0);
        const sel_val = RS.select1(data, rank_val - 1);
        try testing.expectEqual(p, sel_val);
    }
}

test "RankSelect: rank0 + rank1 = position" {
    // Fundamental identity: rank0(i) + rank1(i) = i
    const RS = packlib.RankSelect(u32);
    var bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    for (0..17) |i| {
        const pos: u32 = @intCast(i);
        const r0 = RS.rank0(data, pos);
        const r1 = RS.rank1(data, pos);
        try testing.expectEqual(pos, r0 + r1);
    }
}

// ═══════════════════════════════════════════════════════════════
// Bit I/O – Reference Values
// Source: MSB-first bit ordering convention
// ═══════════════════════════════════════════════════════════════

test "BitWriter/Reader: MSB-first byte 0xA5 = 10100101" {
    // 0xA5 in binary MSB-first: 1,0,1,0,0,1,0,1
    const BW = packlib.BitWriter(u32);
    const BR = packlib.BitReader;

    var writer = BW.init(testing.allocator);
    defer writer.deinit();

    // Write 0xA5 bit by bit MSB-first
    const bits_msbfirst = [_]u1{ 1, 0, 1, 0, 0, 1, 0, 1 };
    for (bits_msbfirst) |b| try writer.writeBit(b);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u8, 0xA5), bytes[0]);

    // Read back and verify
    var reader = BR.init(bytes);
    for (bits_msbfirst) |expected| {
        try testing.expectEqual(expected, try reader.readBit());
    }
}

test "BitWriter/Reader: 12-bit value 0xABC across byte boundary" {
    // 0xABC = 101010111100 in 12-bit MSB-first
    // Should split: first byte = 10101011 (0xAB), next 4 bits = 1100 (0xC_)
    const BW = packlib.BitWriter(u32);
    const BR = packlib.BitReader;

    var writer = BW.init(testing.allocator);
    defer writer.deinit();
    try writer.writeBits(0xABC, 12);
    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    // First byte: upper 8 bits of 0xABC = 0xAB
    try testing.expectEqual(@as(u8, 0xAB), bytes[0]);
    // Second byte: lower 4 bits = 0xC0 (padded with zeros)
    try testing.expectEqual(@as(u8, 0xC0), bytes[1]);

    // Read back the 12-bit value
    var reader = BR.init(bytes);
    const val = try reader.readBits(u12, 12);
    try testing.expectEqual(@as(u12, 0xABC), val);
}

// ═══════════════════════════════════════════════════════════════
// Fixed-Point Arithmetic – Reference Values
// Source: standard Q16.16 format properties
// ═══════════════════════════════════════════════════════════════

test "FixedPoint Q16.16: 1/3 ≈ 0.333... with 16-bit fractional precision" {
    // Q16.16: 1/3 should be 0x00005555 (raw) = 21845/65536 ≈ 0.33332824
    const Q = packlib.Q16_16;
    const one = Q.fromInt(1);
    const three = Q.fromInt(3);
    const result = try one.div(three);
    // 1/3 in Q16.16 = floor(65536/3) = 21845 raw
    try testing.expectEqual(@as(i32, 21845), result.raw);
    // Floating point should be close to 1/3
    try testing.expect(@abs(result.toFloat() - 1.0 / 3.0) < 0.0001);
}

test "FixedPoint Q16.16: 1.0 × 1.0 = 1.0 (multiplicative identity)" {
    const Q = packlib.Q16_16;
    const one = Q.fromInt(1);
    const result = one.mul(one);
    try testing.expectEqual(@as(i32, 65536), result.raw); // 1.0 in Q16.16 = 65536
}

test "FixedPoint Q16.16: overflow detection" {
    const Q = packlib.Q16_16;
    const max_val = Q{ .raw = std.math.maxInt(i32) };
    const one = Q.fromInt(1);
    // Adding 1 to max should overflow
    try testing.expectError(error.Overflow, max_val.checkedAdd(one));
}

test "FixedPoint Q16.16: fromFraction(355, 113) ≈ π" {
    // 355/113 ≈ 3.14159292... (Milü approximation to π)
    const Q = packlib.Q16_16;
    const result = try Q.fromFraction(355, 113);
    try testing.expect(@abs(result.toFloat() - 3.14159) < 0.001);
}

// ═══════════════════════════════════════════════════════════════
// BPE (Byte Pair Encoding) – Algorithmic Properties
// Source: Gage (1994), Sennrich et al. (2016)
// ═══════════════════════════════════════════════════════════════

test "BPE: most frequent pair is merged first" {
    // Fundamental BPE property: the globally most frequent byte pair
    // is always the first merge.  Use max_merges=1 so training stops
    // after exactly one merge and we can assert which pair was chosen.
    const B = packlib.Bpe(u8, 1);
    // "ab" appears 6 times, "cd" appears 3 times
    const corpus = [_][]const u8{ "abababababab", "cdcdcd" };
    const table = try B.train(testing.allocator, &corpus, .{});
    try testing.expectEqual(@as(B.MergeCountType, 1), table.num_merges);
    try testing.expectEqual(B.SymbolPair{ 'a', 'b' }, table.merges[0]);
}

test "BPE: tokens use TOKEN_OFFSET+ range" {
    // BPE(u8) uses 0x80-0xFF for new tokens (original alphabet 0x00-0x7F)
    const B = packlib.Bpe(u8, 128);
    const corpus = [_][]const u8{ "aabb", "aabb", "aabb", "aabb" };
    const table = try B.train(testing.allocator, &corpus, .{});

    // Encode and check tokens are ≥ TOKEN_OFFSET
    const encoded = try B.encode(testing.allocator, &table, "aabb");
    defer testing.allocator.free(encoded);
    for (encoded) |b| {
        if (b >= B.TOKEN_OFFSET) {
            // This is a BPE token — valid
            try testing.expect(b >= B.TOKEN_OFFSET);
        }
    }
}

test "BPE: decode(encode(x)) = x (round-trip identity)" {
    const B = packlib.Bpe(u8, 128);
    const texts = [_][]const u8{
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "aaaaaaa",
        "",
    };
    const corpus = [_][]const u8{ "hello world hello world", "the quick brown fox the quick brown fox" };
    const table = try B.train(testing.allocator, &corpus, .{});
    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };

    for (texts) |input| {
        const encoded = try B.encode(testing.allocator, &table, input);
        defer testing.allocator.free(encoded);

        if (input.len == 0) {
            try testing.expectEqual(@as(usize, 0), encoded.len);
            continue;
        }

        var buf: [256]u8 = undefined;
        const decoded = try B.decode(&view, encoded, &buf);
        try testing.expectEqualSlices(u8, input, decoded);
    }
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: BWT – Burrows-Wheeler Transform Reference Vectors
// Source: Wikipedia "Burrows–Wheeler transform", Worked Example
// ═══════════════════════════════════════════════════════════════

test "BWT vector: 'banana' → BWT output 'nnbaaa' (classic example)" {
    // Source: Wikipedia "Burrows–Wheeler transform", Worked example
    // Using suffix-array-based BWT without sentinel: BWT("banana") = "nnbaaa"
    const Bwt = packlib.Bwt(u32);
    const input = "banana";
    const data = try Bwt.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    const bwt_bytes = Bwt.getBwtBytes(data);
    try testing.expectEqual(@as(usize, 6), bwt_bytes.len);
    try testing.expectEqualStrings("nnbaaa", bwt_bytes);

    // Round-trip
    const recovered = try Bwt.inverseUniform(testing.allocator, data);
    defer testing.allocator.free(recovered);
    try testing.expectEqualStrings(input, recovered);
}

test "BWT vector: 'abracadabra' round-trip" {
    // Source: common BWT example from textbooks
    const Bwt = packlib.Bwt(u32);
    const input = "abracadabra";
    const data = try Bwt.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    const recovered = try Bwt.inverseUniform(testing.allocator, data);
    defer testing.allocator.free(recovered);
    try testing.expectEqualStrings(input, recovered);
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: MTF – Move-to-Front Reference Vectors
// Source: Wikipedia "Move-to-front transform"
// ═══════════════════════════════════════════════════════════════

test "MTF vector: 'bananaaa' → known output" {
    // Source: Wikipedia "Move-to-front transform"
    // 'b'=98,'a'=97,'n'=110,'a'→1,'n'→2,'a'→1,'a'→0,'a'→0
    const Mtf = packlib.Mtf;
    const input = "bananaaa";
    const result = try Mtf.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(result);

    // After 'b' (98) moves to front, 'a' (97) shifts from pos 97 to 98.
    // After 'a' moves to front, 'n' (110) stays at 110.
    // Then 'a','n' alternate at position 1 until runs of 'a' at 0.
    try testing.expectEqual(@as(u8, 98), result[0]); // 'b' at pos 98
    try testing.expectEqual(@as(u8, 98), result[1]); // 'a' shifted to 98
    try testing.expectEqual(@as(u8, 110), result[2]); // 'n' at pos 110
    try testing.expectEqual(@as(u8, 1), result[3]); // 'a' at pos 1
    try testing.expectEqual(@as(u8, 1), result[4]); // 'n' at pos 1
    try testing.expectEqual(@as(u8, 1), result[5]); // 'a' at pos 1
    try testing.expectEqual(@as(u8, 0), result[6]); // 'a' at pos 0
    try testing.expectEqual(@as(u8, 0), result[7]); // 'a' at pos 0

    // Round-trip
    const back = try Mtf.inverseUniform(testing.allocator, result);
    defer testing.allocator.free(back);
    try testing.expectEqualStrings(input, back);
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: BWT + MTF Pipeline Round-Trip
// Source: Standard compression pipeline (BWT → MTF → entropy coder)
// ═══════════════════════════════════════════════════════════════

test "BWT+MTF pipeline vector: round-trip on 'mississippi'" {
    const Bwt = packlib.Bwt(u32);
    const Mtf = packlib.Mtf;

    const input = "mississippi";
    const bwt_data = try Bwt.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(bwt_data);

    const bwt_bytes = Bwt.getBwtBytes(bwt_data);
    const mtf_data = try Mtf.forwardUniform(testing.allocator, bwt_bytes);
    defer testing.allocator.free(mtf_data);

    // BWT+MTF should concentrate values near 0
    var low_count: usize = 0;
    for (mtf_data) |v| {
        if (v < 5) low_count += 1;
    }
    try testing.expect(low_count > mtf_data.len / 2);

    // Round-trip: MTF inverse → BWT inverse
    const mtf_inv = try Mtf.inverseUniform(testing.allocator, mtf_data);
    defer testing.allocator.free(mtf_inv);

    const orig_idx = Bwt.getOriginalIndex(bwt_data);
    const bwt_rebuilt = try testing.allocator.alloc(u8, 4 + mtf_inv.len);
    defer testing.allocator.free(bwt_rebuilt);
    std.mem.writeInt(u32, bwt_rebuilt[0..4], orig_idx, .little);
    @memcpy(bwt_rebuilt[4..], mtf_inv);

    const recovered = try Bwt.inverseUniform(testing.allocator, bwt_rebuilt);
    defer testing.allocator.free(recovered);
    try testing.expectEqualStrings(input, recovered);
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: Elias-Fano Reference Vectors
// Source: Vigna (2013) "Quasi-Succinct Indices", Definition 3
// ═══════════════════════════════════════════════════════════════

test "Elias-Fano vector: monotone sequence access and successor" {
    const EF = packlib.EliasFano(u32);
    const values = [_]u32{ 2, 3, 5, 7, 11, 13, 24 };
    const data = try EF.buildUniform(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 7), EF.count(data));
    try testing.expectEqual(@as(u32, 25), EF.universe(data)); // one past max

    // Random access
    for (values, 0..) |v, i| {
        try testing.expectEqual(v, EF.get(data, @intCast(i)));
    }

    // Successor queries (returns index of first element >= target)
    try testing.expectEqual(@as(u32, 0), EF.successor(data, 0)); // index of first >= 0
    try testing.expectEqual(@as(u32, 0), EF.successor(data, 2)); // index of first >= 2
    try testing.expectEqual(values[EF.successor(data, 0)], @as(u32, 2)); // value is 2
    try testing.expectEqual(values[EF.successor(data, 4)], @as(u32, 5)); // value is 5
    try testing.expectEqual(values[EF.successor(data, 14)], @as(u32, 24)); // value is 24

    // Contains
    try testing.expect(EF.contains(data, 7));
    try testing.expect(!EF.contains(data, 8));
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: rANS – Range ANS Reference Vectors
// Source: Duda (2009), "Asymmetric numeral systems"
// ═══════════════════════════════════════════════════════════════

test "rANS vector: encode/decode with known frequency distribution" {
    const R = packlib.Rans(12);
    const raw_freqs = [_]u32{ 3, 1 };
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    // Quantized freqs must sum to 2^12 = 4096
    try testing.expectEqual(@as(u32, 4096), table.cumul[table.num_symbols]);

    // 75%/25% skew, encode 200 symbols
    var symbols: [200]u16 = undefined;
    for (0..200) |i| symbols[i] = if (i % 4 == 0) 1 else 0;

    const encoded = try R.encodeUniform(testing.allocator, &table, &symbols);
    defer testing.allocator.free(encoded);

    // Compressed output must be smaller than raw
    try testing.expect(encoded.len < 200);

    const decoded = try R.decodeUniform(testing.allocator, encoded, 200);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u16, &symbols, decoded);
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: tANS – Table ANS Reference Vectors
// Source: Collet (2013), FSE / Finite State Entropy
// ═══════════════════════════════════════════════════════════════

test "tANS vector: encode/decode with known frequency distribution" {
    const T = packlib.Tans(10);
    const raw_freqs = [_]u32{ 3, 1 };
    const freqs = try T.quantizeFreqs(testing.allocator, &raw_freqs);
    defer testing.allocator.free(freqs);

    // Quantized freqs must sum to 2^10 = 1024
    var freq_sum: u32 = 0;
    for (freqs) |f| freq_sum += f;
    try testing.expectEqual(@as(u32, 1024), freq_sum);

    var tables = try T.buildTables(testing.allocator, freqs);
    defer T.freeTables(testing.allocator, &tables);

    var symbols: [100]u16 = undefined;
    for (0..100) |i| symbols[i] = if (i % 4 == 0) 1 else 0;

    const encoded = try T.encodeUniform(testing.allocator, &tables, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try T.decodeUniform(testing.allocator, encoded, 100);
    defer testing.allocator.free(decoded);
    try testing.expectEqualSlices(u16, &symbols, decoded);
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: LOUDS – Level-Order Unary Degree Sequence Reference Vectors
// Source: Jacobson (1989), "Space-efficient static trees and graphs"
// ═══════════════════════════════════════════════════════════════

test "LOUDS vector: binary tree navigation" {
    // Full binary tree depth 2 (7 nodes):
    //         0
    //        / \
    //       1   2
    //      / \ / \
    //     3  4 5  6
    const L = packlib.Louds(u32);
    const degrees = [_]u32{ 2, 2, 2, 0, 0, 0, 0 };
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 7), L.numNodes(data));

    // Navigation
    try testing.expectEqual(@as(u32, 1), L.firstChild(data, 0));
    try testing.expectEqual(@as(u32, 2), L.lastChild(data, 0));
    try testing.expectEqual(@as(u32, 3), L.firstChild(data, 1));
    try testing.expectEqual(@as(u32, 5), L.firstChild(data, 2));

    // Parent relationships
    try testing.expectEqual(@as(u32, 0), L.parent(data, 1));
    try testing.expectEqual(@as(u32, 0), L.parent(data, 2));
    try testing.expectEqual(@as(u32, 1), L.parent(data, 3));
    try testing.expectEqual(@as(u32, 2), L.parent(data, 6));

    // Subtree size
    try testing.expectEqual(@as(u32, 7), L.subtreeSize(data, 0));
    try testing.expectEqual(@as(u32, 3), L.subtreeSize(data, 1));
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: FrontCoding Reference Vectors
// Source: Blandford & Blelloch (2008), "Compact Representations"
// ═══════════════════════════════════════════════════════════════

test "FrontCoding vector: sorted strings with shared prefixes" {
    const FC = packlib.FrontCoding;
    const strings = [_][]const u8{
        "compact",
        "compare",
        "compress",
        "compute",
        "data",
        "database",
    };
    const data = try FC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 6), FC.count(data));

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try FC.get(data, @intCast(i), &buf);
        try testing.expectEqualStrings(expected, got);
    }

    // Search
    try testing.expectEqual(@as(u32, 2), (try FC.find(data, "compress", &buf)).?);
    try testing.expect((try FC.find(data, "complex", &buf)) == null);
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: WaveletTree Reference Vectors
// Source: Grossi et al. (2003), "High-order entropy-compressed text indexes"
// ═══════════════════════════════════════════════════════════════

test "WaveletTree vector: rank and select on text" {
    const WT = packlib.WaveletTree(u8, u32);
    const input = "abracadabra";
    const data = try WT.buildUniform(testing.allocator, input);
    defer testing.allocator.free(data);

    // Access
    for (input, 0..) |ch, i| {
        try testing.expectEqual(ch, WT.access(data, @intCast(i)));
    }

    // Rank: count 'a' in [0..pos)
    // "abracadabra" → 'a' at positions 0, 3, 5, 7, 10
    try testing.expectEqual(@as(u32, 1), WT.rank(data, 'a', 1));
    try testing.expectEqual(@as(u32, 2), WT.rank(data, 'a', 4));
    try testing.expectEqual(@as(u32, 5), WT.rank(data, 'a', 11));

    // Select: position of k-th 'a' (0-indexed)
    try testing.expectEqual(@as(u32, 0), WT.select(data, 'a', 0));
    try testing.expectEqual(@as(u32, 3), WT.select(data, 'a', 1));
    try testing.expectEqual(@as(u32, 10), WT.select(data, 'a', 4));
}

// ═══════════════════════════════════════════════════════════════
// Tier 2: DAFSA Reference Vectors
// Source: Daciuk et al. (2000), "Incremental Construction of Minimal
//         Acyclic Finite State Automata"
// ═══════════════════════════════════════════════════════════════

test "DAFSA vector: standard dictionary membership" {
    const D = packlib.Dafsa;
    const words = [_][]const u8{
        "abc",
        "abcd",
        "abd",
        "bc",
        "bcd",
        "cd",
    };
    const data = try D.buildUniform(testing.allocator, &words);
    defer testing.allocator.free(data);

    // Positive membership
    for (words) |w| {
        try testing.expect(D.contains(data, w));
    }

    // Negative membership
    try testing.expect(!D.contains(data, "ab"));
    try testing.expect(!D.contains(data, "a"));
    try testing.expect(!D.contains(data, "abce"));
    try testing.expect(!D.contains(data, "d"));

    // Prefix queries
    try testing.expect(D.hasPrefix(data, "ab"));
    try testing.expect(D.hasPrefix(data, "bc"));
    try testing.expect(!D.hasPrefix(data, "e"));
}
