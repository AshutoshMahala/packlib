//! Static canonical Huffman coding with length-limited codes.
//!
//! `HuffmanCodec(SymbolType)` is generic over the symbol alphabet:
//!  - `u8`  — bytes (256 symbols, standard use case)
//!  - `u4`  — nibbles (16 symbols, e.g. post-MTF with small alphabets)
//!  - `u16` — wide symbols (e.g. BPE merged tokens, up to 65 536 symbols)
//!
//! Code lengths are limited to `MAX_CODE_LEN` (15 for ≤u8, 24 for u16).
//! When the natural Huffman tree exceeds this limit, a Kraft-aware
//! redistribution pass produces valid length-limited codes.
//!
//! Build-time: construct code table from symbol frequencies, encode symbol streams.
//! Read-time: decode symbols from bit stream using serialized table. Zero allocation.
//!
//! Uses canonical Huffman codes for compact table representation (store only code lengths).
//!
//! Serialized table format (via BitWriter):
//!   [16 bits: max_sym]           — highest symbol value + 1 with non-zero frequency
//!   [max_sym × CODE_LEN_BITS bits: code_lengths] — per-symbol code length, ordered by value

const std = @import("std");
const BitReader = @import("../io/bit_reader.zig").BitReader(u32);
const BitWriter = @import("../io/bit_writer.zig").BitWriter;
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn HuffmanCodec(comptime SymbolType: type) type {
    comptime {
        const info = @typeInfo(SymbolType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("HuffmanCodec: SymbolType must be an unsigned integer type, got " ++ @typeName(SymbolType));
        if (@bitSizeOf(SymbolType) < 1)
            @compileError("HuffmanCodec: SymbolType must be at least 1 bit wide");
        if (@bitSizeOf(SymbolType) > 16)
            @compileError("HuffmanCodec: SymbolType wider than u16 is not supported");
    }

    const symbol_bits = @bitSizeOf(SymbolType);
    const alphabet_size: usize = @as(usize, 1) << symbol_bits;

    return struct {
        pub const ALPHABET_SIZE = alphabet_size;

        /// Maximum code length, derived from symbol type.
        /// u1–u8: 15 (matches DEFLATE convention, sufficient for ≤256 symbols).
        /// u9–u16: 24 (supports full u16 alphabets with headroom).
        pub const MAX_CODE_LEN: u8 = if (symbol_bits <= 8) 15 else 24;

        /// Bits needed to serialize a single code-length value.
        pub const CODE_LEN_BITS: u8 = if (MAX_CODE_LEN <= 15) 4 else 5;

        pub const HuffmanTable = struct {
            /// Code length for each symbol. 0 means symbol not in alphabet.
            code_lengths: [ALPHABET_SIZE]u8,
            /// The canonical code for each symbol.
            codes: [ALPHABET_SIZE]u32,
            /// Number of symbols with non-zero code length.
            num_symbols: u32,
        };

        /// Lightweight view for decoding from serialized data. No allocation.
        pub const TableView = struct {
            /// Code lengths indexed by symbol value.
            code_lengths: [ALPHABET_SIZE]u8,
            /// Canonical codes indexed by symbol value.
            codes: [ALPHABET_SIZE]u32,
            num_symbols: u32,
            max_code_len: u8,
            /// Symbols sorted by (code_length, symbol_value) for canonical decode.
            sorted_symbols: [ALPHABET_SIZE]SymbolType,
            /// Count of symbols at each code length (indexed 0..MAX_CODE_LEN).
            sym_count: [MAX_CODE_LEN + 1]u32,
        };

        // ── Build-time ──

        /// Build a Huffman code table using ArenaConfig.
        /// Temp allocations (heap, priority queue) use `arena.temp`.
        pub fn buildTableArena(arena: ArenaConfig, frequencies: []const u64) !HuffmanTable {
            return buildTableImpl(arena.temp, frequencies);
        }

        /// Build a Huffman code table from symbol frequencies.
        /// `frequencies` is indexed by symbol value; freq[i] = frequency of symbol i.
        /// Only the first `ALPHABET_SIZE` entries are used; longer slices are clamped.
        pub fn buildTable(allocator: std.mem.Allocator, frequencies: []const u64) !HuffmanTable {
            return buildTableImpl(allocator, frequencies);
        }

        fn buildTableImpl(allocator: std.mem.Allocator, frequencies: []const u64) !HuffmanTable {
            const num_syms: u32 = @intCast(@min(frequencies.len, ALPHABET_SIZE));

            // Count active symbols.
            var active_count: u32 = 0;
            for (0..num_syms) |i| {
                if (frequencies[i] > 0) active_count += 1;
            }

            if (active_count == 0) {
                return HuffmanTable{
                    .code_lengths = [_]u8{0} ** ALPHABET_SIZE,
                    .codes = [_]u32{0} ** ALPHABET_SIZE,
                    .num_symbols = 0,
                };
            }

            if (active_count == 1) {
                var table = HuffmanTable{
                    .code_lengths = [_]u8{0} ** ALPHABET_SIZE,
                    .codes = [_]u32{0} ** ALPHABET_SIZE,
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

            // ── Phase 1: Build Huffman tree with binary min-heap ──

            const max_nodes: usize = @as(usize, active_count) * 2;
            var nodes = try allocator.alloc(Node, max_nodes);
            defer allocator.free(nodes);
            var node_count: u32 = 0;

            var heap_buf = try allocator.alloc(u32, active_count);
            defer allocator.free(heap_buf);

            for (0..num_syms) |i| {
                if (frequencies[i] > 0) {
                    nodes[node_count] = .{
                        .freq = frequencies[i],
                        .symbol = @intCast(i),
                        .is_leaf = true,
                        .left = 0,
                        .right = 0,
                    };
                    heap_buf[node_count] = node_count;
                    node_count += 1;
                }
            }

            // Build bottom-up min-heap.
            var heap_len: u32 = node_count;
            {
                var i: u32 = heap_len / 2;
                while (i > 0) {
                    i -= 1;
                    heapSiftDown(nodes, heap_buf, heap_len, i);
                }
            }

            // Merge until one root remains.
            while (heap_len > 1) {
                const min1 = heapExtractMin(nodes, heap_buf, &heap_len);
                const min2 = heapExtractMin(nodes, heap_buf, &heap_len);

                nodes[node_count] = .{
                    .freq = nodes[min1].freq + nodes[min2].freq,
                    .symbol = 0,
                    .is_leaf = false,
                    .left = min1,
                    .right = min2,
                };
                heapInsert(nodes, heap_buf, &heap_len, node_count);
                node_count += 1;
            }

            // ── Phase 2: Walk tree to assign initial code lengths (no depth limit) ──

            var code_lengths = [_]u8{0} ** ALPHABET_SIZE;
            const root = heap_buf[0];

            const StackEntry = struct { node_idx: u32, depth: u32 };
            var stack = try allocator.alloc(StackEntry, max_nodes);
            defer allocator.free(stack);
            var stack_len: u32 = 1;
            stack[0] = .{ .node_idx = root, .depth = 0 };

            while (stack_len > 0) {
                stack_len -= 1;
                const item = stack[stack_len];
                const node = nodes[item.node_idx];

                if (node.is_leaf) {
                    // Cap at 255 (u8 max); enforceLengthLimit handles the rest.
                    const depth_u8: u8 = @intCast(@min(item.depth, 255));
                    code_lengths[node.symbol] = if (depth_u8 == 0) 1 else depth_u8;
                } else {
                    stack[stack_len] = .{ .node_idx = node.left, .depth = item.depth + 1 };
                    stack_len += 1;
                    stack[stack_len] = .{ .node_idx = node.right, .depth = item.depth + 1 };
                    stack_len += 1;
                }
            }

            // ── Phase 3: Enforce MAX_CODE_LEN via Kraft-aware redistribution ──

            try enforceLengthLimit(allocator, &code_lengths, active_count, frequencies, num_syms);

            // ── Phase 4: Assign canonical codes ──

            var table = HuffmanTable{
                .code_lengths = code_lengths,
                .codes = [_]u32{0} ** ALPHABET_SIZE,
                .num_symbols = active_count,
            };
            assignCanonicalCodes(&table);

            return table;
        }

        /// Encode a symbol stream using the given table, writing to a BitWriter.
        pub fn encode(table: *const HuffmanTable, symbols: []const SymbolType, writer: *BitWriter(u32)) !void {
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
            for (0..ALPHABET_SIZE) |i| {
                if (table.code_lengths[i] > 0) max_sym = @intCast(i + 1);
            }

            // Write max_sym (16 bits), then per-symbol code lengths.
            try writer.writeBits(@as(u64, max_sym), 16);

            for (0..max_sym) |i| {
                try writer.writeBits(@as(u64, table.code_lengths[i]), CODE_LEN_BITS);
            }
        }

        /// Deserialize a table view from a BitReader (no allocation).
        pub fn deserializeTable(reader: *BitReader) !TableView {
            const max_sym = try reader.readBits(u16, 16);
            if (max_sym > ALPHABET_SIZE) return Error.InvalidCode;

            var view = TableView{
                .code_lengths = [_]u8{0} ** ALPHABET_SIZE,
                .codes = [_]u32{0} ** ALPHABET_SIZE,
                .num_symbols = 0,
                .max_code_len = 0,
                .sorted_symbols = std.mem.zeroes([ALPHABET_SIZE]SymbolType),
                .sym_count = std.mem.zeroes([MAX_CODE_LEN + 1]u32),
            };

            for (0..max_sym) |i| {
                view.code_lengths[i] = try reader.readBits(u8, CODE_LEN_BITS);
                if (view.code_lengths[i] > MAX_CODE_LEN) return Error.InvalidCode;
                if (view.code_lengths[i] > 0) {
                    view.num_symbols += 1;
                    if (view.code_lengths[i] > view.max_code_len) {
                        view.max_code_len = view.code_lengths[i];
                    }
                }
            }

            // Kraft inequality check: reject oversubscribed code-length distributions.
            // Valid prefix codes must satisfy: Σ 2^(MAX_CODE_LEN - l_i) ≤ 2^MAX_CODE_LEN.
            if (view.num_symbols > 0) {
                var kraft_sum: u64 = 0;
                for (0..max_sym) |i| {
                    if (view.code_lengths[i] > 0) {
                        kraft_sum += @as(u64, 1) << @intCast(MAX_CODE_LEN - view.code_lengths[i]);
                    }
                }
                const kraft_total: u64 = @as(u64, 1) << MAX_CODE_LEN;
                if (kraft_sum > kraft_total) return Error.InvalidCode;
            }

            // Reconstruct canonical codes and populate decode structures.
            assignCanonicalCodesView(&view);
            return view;
        }

        /// Decode a single symbol from the bit stream using a TableView.
        /// Uses canonical first-code comparison: O(MAX_CODE_LEN) per symbol.
        pub fn decodeSymbol(table: *const TableView, reader: *BitReader) !SymbolType {
            var code: u32 = 0;
            var first_code: u32 = 0;
            var sym_idx: u32 = 0;

            for (1..@as(usize, table.max_code_len) + 1) |l_usize| {
                const l: u8 = @intCast(l_usize);
                const bit = try reader.readBit();
                code = (code << 1) | @as(u32, bit);

                const count = table.sym_count[l];
                if (count > 0 and code >= first_code and code - first_code < count) {
                    return table.sorted_symbols[@intCast(sym_idx + (code - first_code))];
                }
                first_code = (first_code + count) << 1;
                sym_idx += count;
            }

            return error.InvalidCode;
        }

        // ── Internal types ──

        const Node = struct {
            freq: u64,
            symbol: u32,
            is_leaf: bool,
            left: u32,
            right: u32,
        };

        // ── Binary min-heap over node indices ──

        fn heapNodeLess(nodes: []const Node, a: u32, b: u32) bool {
            if (nodes[a].freq != nodes[b].freq) return nodes[a].freq < nodes[b].freq;
            return a < b; // prefer earlier indices (leaves before internal) for determinism
        }

        fn heapSiftDown(nodes: []const Node, heap: []u32, heap_len: u32, start: u32) void {
            var i = start;
            while (true) {
                var smallest = i;
                const left = 2 * i + 1;
                const right = 2 * i + 2;
                if (left < heap_len and heapNodeLess(nodes, heap[left], heap[smallest]))
                    smallest = left;
                if (right < heap_len and heapNodeLess(nodes, heap[right], heap[smallest]))
                    smallest = right;
                if (smallest == i) break;
                const tmp = heap[i];
                heap[i] = heap[smallest];
                heap[smallest] = tmp;
                i = smallest;
            }
        }

        fn heapExtractMin(nodes: []const Node, heap: []u32, heap_len: *u32) u32 {
            const min_val = heap[0];
            heap_len.* -= 1;
            heap[0] = heap[heap_len.*];
            if (heap_len.* > 0) heapSiftDown(nodes, heap, heap_len.*, 0);
            return min_val;
        }

        fn heapInsert(nodes: []const Node, heap: []u32, heap_len: *u32, val: u32) void {
            heap[heap_len.*] = val;
            var i = heap_len.*;
            heap_len.* += 1;
            while (i > 0) {
                const parent = (i - 1) / 2;
                if (heapNodeLess(nodes, heap[i], heap[parent])) {
                    const tmp = heap[i];
                    heap[i] = heap[parent];
                    heap[parent] = tmp;
                    i = parent;
                } else break;
            }
        }

        // ── Length-limiting ──

        /// Enforce MAX_CODE_LEN via Kraft-aware redistribution.
        /// If any code length exceeds MAX_CODE_LEN, clamp and redistribute so
        /// the resulting lengths form a valid prefix code.
        fn enforceLengthLimit(
            allocator: std.mem.Allocator,
            code_lengths: *[ALPHABET_SIZE]u8,
            active_count: u32,
            frequencies: []const u64,
            num_syms: u32,
        ) !void {
            var needs_limiting = false;
            for (code_lengths) |len| {
                if (len > MAX_CODE_LEN) {
                    needs_limiting = true;
                    break;
                }
            }
            if (!needs_limiting) return;

            // Collect active symbols sorted by (code_length ASC, frequency DESC, symbol ASC).
            var sorted = try allocator.alloc(u32, active_count);
            defer allocator.free(sorted);
            var sorted_idx: u32 = 0;
            for (0..ALPHABET_SIZE) |i| {
                if (code_lengths[i] > 0) {
                    sorted[sorted_idx] = @intCast(i);
                    sorted_idx += 1;
                }
            }

            const SortCtx = struct {
                lengths: *const [ALPHABET_SIZE]u8,
                freqs: []const u64,
                nsyms: u32,
            };
            const ctx = SortCtx{ .lengths = code_lengths, .freqs = frequencies, .nsyms = num_syms };
            std.sort.pdq(u32, sorted[0..sorted_idx], ctx, struct {
                fn lessThan(c: SortCtx, a: u32, b: u32) bool {
                    const la = c.lengths[a];
                    const lb = c.lengths[b];
                    if (la != lb) return la < lb;
                    const fa: u64 = if (a < c.nsyms) c.freqs[a] else 0;
                    const fb: u64 = if (b < c.nsyms) c.freqs[b] else 0;
                    if (fa != fb) return fa > fb; // higher freq first within same length
                    return a < b;
                }
            }.lessThan);

            // Count symbols at each depth, clamping > MAX_CODE_LEN.
            var bl_count = std.mem.zeroes([MAX_CODE_LEN + 1]u32);
            for (sorted[0..sorted_idx]) |sym| {
                const len = @min(code_lengths[sym], MAX_CODE_LEN);
                bl_count[len] += 1;
            }

            // Compute Kraft sum in units of 2^(-MAX_CODE_LEN).
            var kraft_sum: u64 = 0;
            for (1..MAX_CODE_LEN + 1) |d| {
                kraft_sum += @as(u64, bl_count[d]) << @intCast(MAX_CODE_LEN - @as(u8, @intCast(d)));
            }
            const kraft_total: u64 = @as(u64, 1) << MAX_CODE_LEN;

            // Fix oversubscription: lengthen shortest codes to free code space.
            while (kraft_sum > kraft_total) {
                for (1..MAX_CODE_LEN) |d| {
                    if (bl_count[d] > 0) {
                        bl_count[d] -= 1;
                        bl_count[d + 1] += 1;
                        kraft_sum -= @as(u64, 1) << @intCast(MAX_CODE_LEN - @as(u8, @intCast(d)) - 1);
                        break;
                    }
                }
            }

            // Tighten: shorten longest codes to improve compression.
            {
                var d: u8 = MAX_CODE_LEN;
                while (d > 1 and kraft_sum < kraft_total) {
                    if (bl_count[d] > 0) {
                        const increase: u64 = @as(u64, 1) << @intCast(MAX_CODE_LEN - d);
                        if (kraft_sum + increase <= kraft_total) {
                            bl_count[d] -= 1;
                            bl_count[d - 1] += 1;
                            kraft_sum += increase;
                            continue;
                        }
                    }
                    d -= 1;
                }
            }

            // Reassign code lengths: shortest depths to the most frequent symbols.
            var si: u32 = 0;
            for (1..MAX_CODE_LEN + 1) |d| {
                for (0..bl_count[d]) |_| {
                    code_lengths[sorted[si]] = @intCast(d);
                    si += 1;
                }
            }
        }

        // ── Canonical code helpers ──

        fn assignCanonicalCodes(table: *HuffmanTable) void {
            // Sort symbols by (code_length, symbol_value)
            // Then assign codes sequentially within each length group.
            var sorted: [ALPHABET_SIZE]u16 = undefined;
            var sorted_len: u32 = 0;
            for (0..ALPHABET_SIZE) |i| {
                if (table.code_lengths[i] > 0) {
                    sorted[sorted_len] = @intCast(i);
                    sorted_len += 1;
                }
            }

            if (sorted_len == 0) return;

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
            var sorted: [ALPHABET_SIZE]u16 = undefined;
            var sorted_len: u32 = 0;
            for (0..ALPHABET_SIZE) |i| {
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

            // Assign canonical codes.
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

            // Populate decode acceleration structures.
            view.sym_count = std.mem.zeroes([MAX_CODE_LEN + 1]u32);
            for (0..sorted_len) |i| {
                view.sorted_symbols[i] = @intCast(sorted[i]);
                view.sym_count[view.code_lengths[sorted[i]]] += 1;
            }
        }

        fn compareLenSym(code_lengths: *const [ALPHABET_SIZE]u8, a: u16, b: u16) bool {
            if (code_lengths[a] != code_lengths[b]) return code_lengths[a] < code_lengths[b];
            return a < b;
        }

        pub const Error = error{InvalidCode};
    };
}

// ── Tests ────────────────────────────────────────────────────
const testing = std.testing;
const BW = BitWriter(u32);

test "HuffmanCodec: build table from frequencies" {
    const HC = HuffmanCodec(u8);
    // 'a'=5, 'b'=3, 'c'=1, 'd'=1
    var freq = [_]u64{0} ** 256;
    freq['a'] = 5;
    freq['b'] = 3;
    freq['c'] = 1;
    freq['d'] = 1;

    const table = try HC.buildTable(testing.allocator, &freq);
    try testing.expectEqual(@as(u32, 4), table.num_symbols);

    // Most frequent symbol ('a') should have shortest code
    try testing.expect(table.code_lengths['a'] <= table.code_lengths['b']);
    try testing.expect(table.code_lengths['a'] <= table.code_lengths['c']);
    try testing.expect(table.code_lengths['a'] <= table.code_lengths['d']);
}

test "HuffmanCodec: canonical property" {
    const HC = HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
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
    const HC = HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
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
    const enc_bytes = try writer.finish();
    defer testing.allocator.free(enc_bytes);

    // Serialize → deserialize round-trip
    var ser_writer = BW.init(testing.allocator);
    defer ser_writer.deinit();
    try HC.serializeTable(&table, &ser_writer);
    const ser_bytes = try ser_writer.finish();
    defer testing.allocator.free(ser_bytes);

    var ser_reader = BitReader.init(ser_bytes);
    const view = try HC.deserializeTable(&ser_reader);

    // Decode
    var reader = BitReader.init(enc_bytes);
    var decoded: [12]u8 = undefined;
    for (0..input.len) |i| {
        decoded[i] = try HC.decodeSymbol(&view, &reader);
    }
    try testing.expectEqualStrings(input, &decoded);
}

test "HuffmanCodec: single symbol" {
    const HC = HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
    freq['x'] = 100;

    const table = try HC.buildTable(testing.allocator, &freq);
    try testing.expectEqual(@as(u32, 1), table.num_symbols);
    try testing.expectEqual(@as(u8, 1), table.code_lengths['x']);
}

test "HuffmanCodec: empty frequencies" {
    const HC = HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;

    const table = try HC.buildTable(testing.allocator, &freq);
    try testing.expectEqual(@as(u32, 0), table.num_symbols);
}

test "HuffmanCodec: table serialization round-trip" {
    const HC = HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
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
    const HC = HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
    freq['a'] = 990;
    freq['b'] = 5;
    freq['c'] = 3;
    freq['d'] = 2;

    const table = try HC.buildTable(testing.allocator, &freq);
    // Dominant symbol should get shortest code (1 bit)
    try testing.expectEqual(@as(u8, 1), table.code_lengths['a']);
}

test "HuffmanCodec: frequency weighting" {
    const HC = HuffmanCodec(u8);
    var freq = [_]u64{0} ** 256;
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

test "HuffmanCodec: deserialize rejects corrupt code lengths" {
    const HC = HuffmanCodec(u8);

    // Craft a bitstream with a code_length > MAX_CODE_LEN (15).
    // Format: [u16: max_sym=1] [4 bits: code_length=15] — valid edge case first.
    {
        var writer = BW.init(testing.allocator);
        defer writer.deinit();
        try writer.writeBits(1, 16); // max_sym = 1
        try writer.writeBits(15, 4); // code_length = 15 (valid, == MAX_CODE_LEN)
        const bytes = try writer.finish();
        defer testing.allocator.free(bytes);
        var reader = BitReader.init(bytes);
        const view = try HC.deserializeTable(&reader);
        try testing.expectEqual(@as(u8, 15), view.code_lengths[0]);
    }
}

test "HuffmanCodec: u4 symbol type (tiny alphabet)" {
    const HC = HuffmanCodec(u4);
    try testing.expectEqual(@as(usize, 16), HC.ALPHABET_SIZE);

    var freq = [_]u64{0} ** 16;
    freq[0] = 10;
    freq[1] = 5;
    freq[2] = 3;
    freq[3] = 1;

    const table = try HC.buildTable(testing.allocator, &freq);
    try testing.expectEqual(@as(u32, 4), table.num_symbols);

    // Encode
    const input = [_]u4{ 0, 0, 1, 0, 2, 0, 3, 0, 0, 1 };
    var writer = BW.init(testing.allocator);
    defer writer.deinit();
    try HC.encode(&table, &input, &writer);
    const enc_bytes = try writer.finish();
    defer testing.allocator.free(enc_bytes);

    // Serialize + deserialize round-trip
    var ser_writer = BW.init(testing.allocator);
    defer ser_writer.deinit();
    try HC.serializeTable(&table, &ser_writer);
    const ser_bytes = try ser_writer.finish();
    defer testing.allocator.free(ser_bytes);

    var ser_reader = BitReader.init(ser_bytes);
    const view = try HC.deserializeTable(&ser_reader);

    // Decode and verify round-trip
    var reader = BitReader.init(enc_bytes);
    for (input) |expected| {
        try testing.expectEqual(expected, try HC.decodeSymbol(&view, &reader));
    }
}

test "HuffmanCodec: u1 binary alphabet" {
    const HC = HuffmanCodec(u1);
    try testing.expectEqual(@as(usize, 2), HC.ALPHABET_SIZE);

    var freq = [_]u64{ 7, 3 };
    const table = try HC.buildTable(testing.allocator, &freq);
    try testing.expectEqual(@as(u32, 2), table.num_symbols);

    // Both symbols must get code length 1 (binary alphabet = 1 bit each)
    try testing.expectEqual(@as(u8, 1), table.code_lengths[0]);
    try testing.expectEqual(@as(u8, 1), table.code_lengths[1]);
}
