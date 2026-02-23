//! Static canonical Huffman coding.
//!
//! Build-time: construct code table from symbol frequencies, encode symbol streams.
//! Read-time: decode symbols from bit stream using serialized table. Zero allocation.
//!
//! Uses canonical Huffman codes for compact table representation (store only code lengths).
//!
//! Serialized table format:
//!   [1 byte: num_symbols - 1 (0 = 1 symbol, 255 = 256 symbols)]
//!   [num_symbols bytes: code_lengths[symbol_0], code_lengths[symbol_1], ...]
//!   Symbols are ordered by their byte value (0x00..0xFF).

const std = @import("std");
const BitReader = @import("../io/bit_reader.zig").BitReader;
const BitWriter = @import("../io/bit_writer.zig").BitWriter;

pub fn HuffmanCodec(comptime FreqType: type) type {
    return struct {
        const Self = @This();

        pub const MAX_SYMBOLS = 256;
        pub const MAX_CODE_LEN = 15; // practical limit for canonical Huffman

        pub const HuffmanTable = struct {
            /// Code length for each symbol. 0 means symbol not in alphabet.
            code_lengths: [MAX_SYMBOLS]u8,
            /// The canonical code for each symbol.
            codes: [MAX_SYMBOLS]u32,
            /// Number of symbols with non-zero code length.
            num_symbols: u16,
        };

        /// Lightweight view for decoding from serialized data. No allocation.
        pub const TableView = struct {
            /// Code lengths indexed by symbol value.
            code_lengths: [MAX_SYMBOLS]u8,
            /// Canonical codes indexed by symbol value.
            codes: [MAX_SYMBOLS]u32,
            num_symbols: u16,
            max_code_len: u8,
        };

        // ── Build-time ──

        /// Build a Huffman code table from symbol frequencies.
        /// `frequencies` is indexed by symbol value; freq[i] = frequency of symbol i.
        pub fn buildTable(allocator: std.mem.Allocator, frequencies: []const FreqType) !HuffmanTable {
            const num_syms: u16 = @intCast(@min(frequencies.len, MAX_SYMBOLS));

            // Count active symbols
            var active_count: u16 = 0;
            for (0..num_syms) |i| {
                if (frequencies[i] > 0) active_count += 1;
            }

            if (active_count == 0) {
                return HuffmanTable{
                    .code_lengths = [_]u8{0} ** MAX_SYMBOLS,
                    .codes = [_]u32{0} ** MAX_SYMBOLS,
                    .num_symbols = 0,
                };
            }

            if (active_count == 1) {
                // Single symbol: assign code length 1, code 0
                var table = HuffmanTable{
                    .code_lengths = [_]u8{0} ** MAX_SYMBOLS,
                    .codes = [_]u32{0} ** MAX_SYMBOLS,
                    .num_symbols = 1,
                };
                for (0..num_syms) |i| {
                    if (frequencies[i] > 0) {
                        table.code_lengths[i] = 1;
                        table.codes[i] = 0;
                        break;
                    }
                }
                return table;
            }

            // Build Huffman tree using a min-heap priority queue.
            // Each node has a frequency and optionally left/right children.
            const Node = struct {
                freq: u64,
                symbol: u16, // valid only for leaf nodes
                is_leaf: bool,
                left: u16, // index into nodes array
                right: u16,
            };

            var nodes = try allocator.alloc(Node, @as(usize, active_count) * 2);
            defer allocator.free(nodes);
            var node_count: u16 = 0;

            // Create leaf nodes
            for (0..num_syms) |i| {
                if (frequencies[i] > 0) {
                    nodes[node_count] = .{
                        .freq = @intCast(frequencies[i]),
                        .symbol = @intCast(i),
                        .is_leaf = true,
                        .left = 0,
                        .right = 0,
                    };
                    node_count += 1;
                }
            }

            // Simple priority queue: use an array and find min each time.
            // For ≤256 symbols this is fast enough.
            var active = try allocator.alloc(u16, node_count);
            defer allocator.free(active);
            var active_len: u16 = node_count;
            for (0..node_count) |i| {
                active[i] = @intCast(i);
            }

            while (active_len > 1) {
                // Find two minimum frequency nodes
                const min1_idx = findMin(nodes, active[0..active_len]);
                const min1 = active[min1_idx];
                active[min1_idx] = active[active_len - 1];
                active_len -= 1;

                const min2_idx = findMin(nodes, active[0..active_len]);
                const min2 = active[min2_idx];
                active[min2_idx] = active[active_len - 1];
                active_len -= 1;

                // Create internal node
                nodes[node_count] = .{
                    .freq = nodes[min1].freq + nodes[min2].freq,
                    .symbol = 0,
                    .is_leaf = false,
                    .left = min1,
                    .right = min2,
                };
                active[active_len] = node_count;
                active_len += 1;
                node_count += 1;
            }

            // Walk tree to assign code lengths
            var code_lengths = [_]u8{0} ** MAX_SYMBOLS;
            const root = active[0];

            var stack: [MAX_CODE_LEN * 2 + 2]struct { node_idx: u16, depth: u8 } = undefined;
            var stack_len: u32 = 0;
            stack[0] = .{ .node_idx = root, .depth = 0 };
            stack_len = 1;

            while (stack_len > 0) {
                stack_len -= 1;
                const item = stack[stack_len];
                const node = nodes[item.node_idx];

                if (node.is_leaf) {
                    code_lengths[node.symbol] = if (item.depth == 0) 1 else item.depth;
                } else {
                    // Clamp depth to avoid overflow
                    const next_depth = if (item.depth < MAX_CODE_LEN) item.depth + 1 else MAX_CODE_LEN;
                    stack[stack_len] = .{ .node_idx = node.left, .depth = next_depth };
                    stack_len += 1;
                    stack[stack_len] = .{ .node_idx = node.right, .depth = next_depth };
                    stack_len += 1;
                }
            }

            // Assign canonical codes
            var table = HuffmanTable{
                .code_lengths = code_lengths,
                .codes = [_]u32{0} ** MAX_SYMBOLS,
                .num_symbols = active_count,
            };
            assignCanonicalCodes(&table);

            return table;
        }

        /// Encode a symbol stream using the given table, writing to a BitWriter.
        pub fn encode(table: *const HuffmanTable, symbols: []const u8, writer: *BitWriter(u32)) !void {
            for (symbols) |sym| {
                const len = table.code_lengths[sym];
                if (len == 0) continue; // symbol not in table
                try writer.writeBits(@as(u64, table.codes[sym]), @intCast(len));
            }
        }

        /// Serialize the code table for storage.
        pub fn serializeTable(table: *const HuffmanTable, writer: *BitWriter(u32)) !void {
            // Find active symbol range
            var max_sym: u16 = 0;
            for (0..MAX_SYMBOLS) |i| {
                if (table.code_lengths[i] > 0) max_sym = @intCast(i + 1);
            }

            // Write num_symbols (as max symbol value, so we know how many lengths follow)
            try writer.writeBits(@as(u64, max_sym), 16);

            // Write code lengths
            for (0..max_sym) |i| {
                try writer.writeBits(@as(u64, table.code_lengths[i]), 4); // max 15 fits in 4 bits
            }
        }

        /// Deserialize a table view from a BitReader (no allocation).
        pub fn deserializeTable(reader: *BitReader) !TableView {
            const max_sym = try reader.readBits(u16, 16);

            var view = TableView{
                .code_lengths = [_]u8{0} ** MAX_SYMBOLS,
                .codes = [_]u32{0} ** MAX_SYMBOLS,
                .num_symbols = 0,
                .max_code_len = 0,
            };

            for (0..max_sym) |i| {
                view.code_lengths[i] = try reader.readBits(u8, 4);
                if (view.code_lengths[i] > 0) {
                    view.num_symbols += 1;
                    if (view.code_lengths[i] > view.max_code_len) {
                        view.max_code_len = view.code_lengths[i];
                    }
                }
            }

            // Reconstruct canonical codes
            assignCanonicalCodesView(&view);
            return view;
        }

        /// Decode a single symbol from the bit stream using a TableView.
        pub fn decodeSymbol(table: *const TableView, reader: *BitReader) !u8 {
            // Bit-by-bit decoding: accumulate bits and check against all codes.
            // For small alphabets (≤256) this is efficient enough.
            var code: u32 = 0;
            var len: u8 = 0;

            while (len < table.max_code_len) {
                const bit = try reader.readBit();
                code = (code << 1) | @as(u32, bit);
                len += 1;

                // Check if any symbol has this code at this length
                for (0..MAX_SYMBOLS) |sym| {
                    if (table.code_lengths[sym] == len and table.codes[sym] == code) {
                        return @intCast(sym);
                    }
                }
            }

            return error.InvalidCode;
        }

        // ── Internal helpers ──

        fn findMin(nodes: anytype, active: []const u16) usize {
            var min_idx: usize = 0;
            var min_freq: u64 = nodes[active[0]].freq;
            for (1..active.len) |i| {
                if (nodes[active[i]].freq < min_freq) {
                    min_freq = nodes[active[i]].freq;
                    min_idx = i;
                }
            }
            return min_idx;
        }

        fn assignCanonicalCodes(table: *HuffmanTable) void {
            // Sort symbols by (code_length, symbol_value)
            // Then assign codes sequentially within each length group.
            var sorted: [MAX_SYMBOLS]u16 = undefined;
            var sorted_len: u16 = 0;
            for (0..MAX_SYMBOLS) |i| {
                if (table.code_lengths[i] > 0) {
                    sorted[sorted_len] = @intCast(i);
                    sorted_len += 1;
                }
            }

            // Insertion sort by (code_length, symbol)
            for (1..sorted_len) |i| {
                const key = sorted[i];
                var j: usize = i;
                while (j > 0 and compareLenSym(&table.code_lengths, key, sorted[j - 1])) {
                    sorted[j] = sorted[j - 1];
                    j -= 1;
                }
                sorted[j] = key;
            }

            // Assign canonical codes
            var code: u32 = 0;
            var prev_len: u8 = table.code_lengths[sorted[0]];

            for (0..sorted_len) |i| {
                const sym = sorted[i];
                const cur_len = table.code_lengths[sym];
                if (cur_len > prev_len) {
                    code <<= @intCast(cur_len - prev_len);
                    prev_len = cur_len;
                }
                table.codes[sym] = code;
                code += 1;
            }
        }

        fn assignCanonicalCodesView(view: *TableView) void {
            var sorted: [MAX_SYMBOLS]u16 = undefined;
            var sorted_len: u16 = 0;
            for (0..MAX_SYMBOLS) |i| {
                if (view.code_lengths[i] > 0) {
                    sorted[sorted_len] = @intCast(i);
                    sorted_len += 1;
                }
            }

            if (sorted_len == 0) return;

            for (1..sorted_len) |i| {
                const key = sorted[i];
                var j: usize = i;
                while (j > 0 and compareLenSym(&view.code_lengths, key, sorted[j - 1])) {
                    sorted[j] = sorted[j - 1];
                    j -= 1;
                }
                sorted[j] = key;
            }

            var code: u32 = 0;
            var prev_len: u8 = view.code_lengths[sorted[0]];

            for (0..sorted_len) |i| {
                const sym = sorted[i];
                const cur_len = view.code_lengths[sym];
                if (cur_len > prev_len) {
                    code <<= @intCast(cur_len - prev_len);
                    prev_len = cur_len;
                }
                view.codes[sym] = code;
                code += 1;
            }
        }

        fn compareLenSym(code_lengths: *const [MAX_SYMBOLS]u8, a: u16, b: u16) bool {
            if (code_lengths[a] != code_lengths[b]) return code_lengths[a] < code_lengths[b];
            return a < b;
        }

        pub const Error = error{ InvalidCode, EndOfStream };
    };
}

// ── Tests ────────────────────────────────────────────────────
const testing = std.testing;
const BW = BitWriter(u32);

test "HuffmanCodec: build table from frequencies" {
    const HC = HuffmanCodec(u32);
    // 'a'=5, 'b'=3, 'c'=1, 'd'=1
    var freq = [_]u32{0} ** 256;
    freq['a'] = 5;
    freq['b'] = 3;
    freq['c'] = 1;
    freq['d'] = 1;

    const table = try HC.buildTable(testing.allocator, &freq);
    try testing.expectEqual(@as(u16, 4), table.num_symbols);

    // Most frequent symbol ('a') should have shortest code
    try testing.expect(table.code_lengths['a'] <= table.code_lengths['b']);
    try testing.expect(table.code_lengths['a'] <= table.code_lengths['c']);
    try testing.expect(table.code_lengths['a'] <= table.code_lengths['d']);
}

test "HuffmanCodec: canonical property" {
    const HC = HuffmanCodec(u32);
    var freq = [_]u32{0} ** 256;
    freq['a'] = 10;
    freq['b'] = 5;
    freq['c'] = 3;
    freq['d'] = 2;
    freq['e'] = 1;

    const table = try HC.buildTable(testing.allocator, &freq);

    // Canonical property: for symbols with the same code length,
    // the symbol with smaller value gets the smaller code.
    for (0..255) |i| {
        for (i + 1..256) |j| {
            if (table.code_lengths[i] > 0 and table.code_lengths[i] == table.code_lengths[j]) {
                try testing.expect(table.codes[i] < table.codes[j]);
            }
        }
    }
}

test "HuffmanCodec: round-trip encode/decode" {
    const HC = HuffmanCodec(u32);
    var freq = [_]u32{0} ** 256;
    freq['a'] = 10;
    freq['b'] = 5;
    freq['c'] = 2;
    freq['d'] = 1;

    const table = try HC.buildTable(testing.allocator, &freq);

    // Encode
    const input = "aabacadaabba";
    var writer = BW.init(testing.allocator);
    defer writer.deinit();
    try HC.encode(&table, input, &writer);
    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    // Deserialize through serialize round-trip first
    var ser_writer = BW.init(testing.allocator);
    defer ser_writer.deinit();
    try HC.serializeTable(&table, &ser_writer);
    const ser_bytes = try ser_writer.finish();
    defer testing.allocator.free(ser_bytes);

    // Re-encode with same table for decode test
    var writer2 = BW.init(testing.allocator);
    defer writer2.deinit();
    try HC.encode(&table, input, &writer2);
    const enc_bytes = try writer2.finish();
    defer testing.allocator.free(enc_bytes);

    // Build view from original table (simulating deserialization)
    var view = HC.TableView{
        .code_lengths = table.code_lengths,
        .codes = table.codes,
        .num_symbols = table.num_symbols,
        .max_code_len = 0,
    };
    for (0..256) |i| {
        if (view.code_lengths[i] > view.max_code_len) {
            view.max_code_len = view.code_lengths[i];
        }
    }

    // Decode
    var reader = BitReader.init(enc_bytes);
    var decoded: [12]u8 = undefined;
    for (0..input.len) |i| {
        decoded[i] = try HC.decodeSymbol(&view, &reader);
    }
    try testing.expectEqualStrings(input, &decoded);
}

test "HuffmanCodec: single symbol" {
    const HC = HuffmanCodec(u32);
    var freq = [_]u32{0} ** 256;
    freq['x'] = 100;

    const table = try HC.buildTable(testing.allocator, &freq);
    try testing.expectEqual(@as(u16, 1), table.num_symbols);
    try testing.expectEqual(@as(u8, 1), table.code_lengths['x']);
}

test "HuffmanCodec: empty frequencies" {
    const HC = HuffmanCodec(u32);
    var freq = [_]u32{0} ** 256;

    const table = try HC.buildTable(testing.allocator, &freq);
    try testing.expectEqual(@as(u16, 0), table.num_symbols);
}

test "HuffmanCodec: table serialization round-trip" {
    const HC = HuffmanCodec(u32);
    var freq = [_]u32{0} ** 256;
    freq['a'] = 10;
    freq['b'] = 5;
    freq['c'] = 2;

    const table = try HC.buildTable(testing.allocator, &freq);

    // Serialize
    var writer = BW.init(testing.allocator);
    defer writer.deinit();
    try HC.serializeTable(&table, &writer);
    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    // Deserialize
    var reader = BitReader.init(bytes);
    const view = try HC.deserializeTable(&reader);

    // Code lengths should match
    for (0..256) |i| {
        try testing.expectEqual(table.code_lengths[i], view.code_lengths[i]);
    }
    try testing.expectEqual(table.num_symbols, view.num_symbols);
}

test "HuffmanCodec: skewed distribution" {
    const HC = HuffmanCodec(u32);
    var freq = [_]u32{0} ** 256;
    freq['a'] = 990;
    freq['b'] = 5;
    freq['c'] = 3;
    freq['d'] = 2;

    const table = try HC.buildTable(testing.allocator, &freq);
    // Dominant symbol should get shortest code (1 bit)
    try testing.expectEqual(@as(u8, 1), table.code_lengths['a']);
}

test "HuffmanCodec: frequency weighting" {
    const HC = HuffmanCodec(u32);
    var freq = [_]u32{0} ** 256;
    freq['a'] = 100;
    freq['b'] = 50;
    freq['c'] = 10;
    freq['d'] = 1;

    const table = try HC.buildTable(testing.allocator, &freq);
    // More frequent → shorter or equal code
    try testing.expect(table.code_lengths['a'] <= table.code_lengths['b']);
    try testing.expect(table.code_lengths['b'] <= table.code_lengths['c']);
    try testing.expect(table.code_lengths['c'] <= table.code_lengths['d']);
}
