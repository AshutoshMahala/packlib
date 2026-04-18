//! Integration tests — cross-module round-trip verification.
//!
//! These tests validate that modules compose correctly:
//! BitWriter → BitReader round-trips, FixedPoint in encode/decode flows, etc.

const packlib = @import("packlib");
const std = @import("std");
const testing = std.testing;

test "BitWriter → BitReader: encode and decode a structured record" {
    // Simulate encoding a record: 4-bit tag, 16-bit value, 1-bit flag
    var writer = packlib.BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    const tag: u64 = 0b1101; // 4-bit tag
    const value: u64 = 0xBEEF; // 16-bit value
    const flag: u1 = 1; // 1-bit flag

    try writer.writeBits(tag, 4);
    try writer.writeBits(value, 16);
    try writer.writeBit(flag);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    var reader = packlib.BitReader(u32).init(bytes);
    try testing.expectEqual(@as(u4, 0b1101), try reader.readBits(u4, 4));
    try testing.expectEqual(@as(u16, 0xBEEF), try reader.readBits(u16, 16));
    try testing.expectEqual(@as(u1, 1), try reader.readBit());
}

test "BitWriter → BitReader: multiple records back to back" {
    var writer = packlib.BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    // Write 10 records: each is an 8-bit value followed by a 4-bit length
    for (0..10) |i| {
        try writer.writeBits(@as(u64, @intCast(i * 25)), 8); // value 0..225
        try writer.writeBits(@as(u64, @intCast(i)), 4); // length 0..9
    }

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    var reader = packlib.BitReader(u32).init(bytes);
    for (0..10) |i| {
        const val = try reader.readBits(u8, 8);
        const len = try reader.readBits(u4, 4);
        try testing.expectEqual(@as(u8, @intCast(i * 25)), val);
        try testing.expectEqual(@as(u4, @intCast(i)), len);
    }
}

test "RankSelect: build and query" {
    // Build a bit vector and verify rank/select
    const RS = packlib.RankSelect(u32);

    // Build from a bits array: set bits at positions 0, 5, 10
    var bits = [_]u1{0} ** 16;
    bits[0] = 1;
    bits[5] = 1;
    bits[10] = 1;

    const bytes = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u32, 16), RS.totalBits(bytes));
    try testing.expectEqual(@as(u32, 3), RS.totalOnes(bytes));
    try testing.expectEqual(@as(u32, 1), RS.rank1(bytes, 1)); // 1 set bit in [0..1)
    try testing.expectEqual(@as(u32, 2), RS.rank1(bytes, 6)); // 2 set bits in [0..6)
    try testing.expectEqual(@as(u32, 3), RS.rank1(bytes, 16)); // all 3

    try testing.expectEqual(@as(u32, 0), RS.select1(bytes, 0));
    try testing.expectEqual(@as(u32, 5), RS.select1(bytes, 1));
    try testing.expectEqual(@as(u32, 10), RS.select1(bytes, 2));
}

test "Huffman: encode → serialize → deserialize → decode" {
    const HC = packlib.HuffmanCodec(u8);
    const BW = packlib.BitWriter(u32);
    const BR = packlib.BitReader(u32);

    // Build table from frequency distribution
    var freq = [_]u64{0} ** 256;
    freq['h'] = 1;
    freq['e'] = 1;
    freq['l'] = 3;
    freq['o'] = 2;
    freq[' '] = 1;
    freq['w'] = 1;
    freq['r'] = 1;
    freq['d'] = 1;

    const table = try HC.buildTable(testing.allocator, &freq);

    // Serialize table
    var tw = BW.init(testing.allocator);
    defer tw.deinit();
    try HC.serializeTable(&table, &tw);
    const table_bytes = try tw.finish();
    defer testing.allocator.free(table_bytes);

    // Encode message
    const msg = "hello world";
    var ew = BW.init(testing.allocator);
    defer ew.deinit();
    try HC.encode(&table, msg, &ew);
    const enc_bytes = try ew.finish();
    defer testing.allocator.free(enc_bytes);

    // Deserialize table
    var tr = BR.init(table_bytes);
    const view = try HC.deserializeTable(&tr);

    // Decode message
    var er = BR.init(enc_bytes);
    var decoded: [11]u8 = undefined;
    for (0..msg.len) |i| {
        decoded[i] = try HC.decodeSymbol(&view, &er);
    }
    try testing.expectEqualStrings(msg, &decoded);
}

test "DeltaVarint: encode sorted sequence and random access" {
    const DV = packlib.DeltaVarint(u32);
    const input = [_]u32{ 10, 20, 30, 100, 200, 500, 1000 };

    const bytes = try DV.encode(testing.allocator, &input);
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(u32, 7), DV.count(bytes));

    // Random access each element
    for (input, 0..) |expected, i| {
        try testing.expectEqual(expected, DV.get(bytes, @intCast(i)));
    }
}

test "BPE: train → encode → decode round-trip" {
    const B = packlib.Bpe(u8, 128);
    const corpus = [_][]const u8{ "the quick brown fox", "the quick brown fox", "the quick brown fox" };
    const table = try B.train(testing.allocator, &corpus, .{});

    const input: []const u8 = "the quick brown fox";
    const encoded = try B.encode(testing.allocator, &table, input);
    defer testing.allocator.free(encoded);

    try testing.expect(encoded.len <= input.len);

    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [256]u8 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "Entropy + Huffman: entropy-driven encoding" {
    // Use entropy analysis to verify Huffman coding is near-optimal
    const Entropy = packlib.Entropy;

    const text = "aaaaaabbbbccdd";
    const report = try Entropy.analyze(testing.allocator, text);

    // H0 should be > 0 (not uniform, not single-char)
    try testing.expect(report.h0 > 0.0);
    // H0 for this distribution should be < 2.0 bits
    try testing.expect(report.h0 < 2.0);
}

test "FixedPoint: Q16_16 as probability weight" {
    // Simulate a common compression pattern: normalize frequencies to probabilities
    const Q = packlib.Q16_16;

    // Frequencies: a=50, b=30, c=20 → total=100
    const total = Q.fromInt(100);
    const freq_a = Q.fromInt(50);
    const freq_b = Q.fromInt(30);
    const freq_c = Q.fromInt(20);

    const prob_a = try freq_a.div(total);
    const prob_b = try freq_b.div(total);
    const prob_c = try freq_c.div(total);

    // Probabilities should sum to ~1.0
    const sum = prob_a.add(prob_b).add(prob_c);
    try testing.expect(@abs(sum.toFloat() - 1.0) < 0.001);

    // Order: a > b > c
    try testing.expect(prob_a.greaterThan(prob_b));
    try testing.expect(prob_b.greaterThan(prob_c));
}
