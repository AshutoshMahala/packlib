//! Byte-pair encoding (BPE) — iterative digram replacement, generic over symbol
//! type and merge budget.
//!
//! `Bpe(SymbolType, max_merges)` operates over sequences of `SymbolType` values.
//! The original alphabet occupies `[0, TOKEN_OFFSET)` and merge tokens occupy
//! `[TOKEN_OFFSET, TOKEN_OFFSET + max_merges)` at the top of the symbol space.
//!
//! Two independent comptime knobs:
//!  - `SymbolType` — controls symbol width / memory layout (u8, u9, u16, …)
//!  - `max_merges` — controls compression depth (how many merge tokens to reserve)
//!
//! Build-time: train a merge table from a corpus, encode symbol streams.
//! Read-time: decode using serialized table + caller-provided buffer. Zero allocation.
//!
//! ## Examples
//!
//! - `Bpe(u8, 128)`: classic byte BPE. 128 ASCII symbols, 128 merges.
//! - `Bpe(u16, 128)`: full byte-range BPE. All 256 byte values as input, 128 merges.
//!   TOKEN_OFFSET = 65408. Use when input contains bytes ≥ 0x80.
//! - `Bpe(u16, 512)`: deep merge BPE for highly repetitive corpora.
//! - `Bpe(u9, 256)`: 256 original symbols + 256 merges in 9 bits. Tight.
//!
//! ## Serialized table format
//!
//! ```text
//! [sizeof(MergeCountType) bytes: num_merges]
//! [num_merges × 2 × sizeof(SymbolType) bytes: pair (left, right) for each merge]
//! ```
//!
//! ## References
//!
//! - Gage (1994), "A New Algorithm for Data Compression"
//! - Sennrich, Haddow, Birch (2016), "Neural Machine Translation of Rare Words
//!   with Subword Units"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn Bpe(comptime SymbolType: type, comptime max_merges: comptime_int) type {
    comptime {
        const info = @typeInfo(SymbolType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("Bpe: SymbolType must be an unsigned integer type, got " ++ @typeName(SymbolType));
        if (@bitSizeOf(SymbolType) < 2)
            @compileError("Bpe: SymbolType must be at least 2 bits wide");
        if (@bitSizeOf(SymbolType) > 16)
            @compileError("Bpe: SymbolType wider than u16 is not supported (pair counting would be prohibitively expensive)");
    }

    const symbol_bits = @bitSizeOf(SymbolType);
    const symbol_count: usize = @as(usize, 1) << symbol_bits;

    comptime {
        if (max_merges < 1)
            @compileError("Bpe: max_merges must be at least 1");
        if (max_merges >= symbol_count)
            @compileError("Bpe: max_merges must be less than the symbol space (2^bits). No room for original alphabet.");
    }

    // Tokens occupy the TOP of the symbol space: [symbol_count - max_merges, symbol_count).
    // Original alphabet occupies [0, symbol_count - max_merges).
    const token_offset_usize: usize = symbol_count - max_merges;

    // Use flat pair-count arrays for small effective symbol counts, HashMap for larger.
    const use_flat = symbol_count <= 256;

    return struct {
        /// Number of distinct symbols representable by SymbolType.
        pub const SYMBOL_COUNT: usize = symbol_count;

        /// First merge token value. Original alphabet is [0, TOKEN_OFFSET).
        pub const TOKEN_OFFSET: SymbolType = @intCast(token_offset_usize);

        /// Maximum number of merges (comptime budget).
        pub const MAX_MERGES: usize = max_merges;

        /// Size of the original alphabet [0, TOKEN_OFFSET).
        pub const ALPHABET_SIZE: usize = token_offset_usize;

        /// Smallest unsigned int that can hold a merge count in [0, MAX_MERGES].
        pub const MergeCountType = std.math.IntFittingRange(0, MAX_MERGES);

        pub const SymbolPair = [2]SymbolType;

        pub const BpeTable = struct {
            /// merges[i] = the pair (left, right) that maps to token TOKEN_OFFSET+i.
            /// left and right can themselves be tokens from earlier merges.
            merges: [MAX_MERGES]SymbolPair,
            /// How many merges were trained.
            num_merges: MergeCountType,
        };

        /// Lightweight read-only view over serialized BPE table.
        pub const TableView = struct {
            merges: [MAX_MERGES]SymbolPair,
            num_merges: MergeCountType,
        };

        /// HashMap key for pair counting (used when symbol_count > 256).
        const PairKey = struct {
            a: SymbolType,
            b: SymbolType,
        };

        const PairContext = struct {
            pub fn hash(_: @This(), key: PairKey) u64 {
                // Combine both symbols into a single integer for hashing.
                const a: u64 = key.a;
                const b: u64 = key.b;
                const combined = (a << symbol_bits) | b;
                return std.hash.Wyhash.hash(0, std.mem.asBytes(&combined));
            }
            pub fn eql(_: @This(), a: PairKey, b: PairKey) bool {
                return a.a == b.a and a.b == b.b;
            }
        };

        const PairMap = std.HashMap(PairKey, u32, PairContext, 80);

        /// Default minimum pair frequency when not specified.
        pub const DEFAULT_MIN_FREQUENCY: u32 = 2;

        // ── Build-time ──

        /// Train a merge table using ArenaConfig. Working buffers use `arena.temp`.
        pub fn trainArena(arena: ArenaConfig, corpus: []const []const SymbolType, options: TrainOptions) !BpeTable {
            return trainImpl(arena.temp, corpus, options);
        }

        /// Train a merge table from a corpus of symbol slices.
        /// Iteratively finds the most frequent adjacent pair and replaces it with a new token.
        pub fn train(allocator: std.mem.Allocator, corpus: []const []const SymbolType, options: TrainOptions) !BpeTable {
            return trainImpl(allocator, corpus, options);
        }

        pub const TrainOptions = struct {
            /// Minimum pair frequency to consider a merge. Pairs occurring fewer
            /// times than this are skipped. Default: 2 (classic BPE behavior).
            min_frequency: u32 = DEFAULT_MIN_FREQUENCY,
        };

        fn trainImpl(allocator: std.mem.Allocator, corpus: []const []const SymbolType, options: TrainOptions) !BpeTable {
            // Concatenate corpus into a mutable working buffer
            var total_len: usize = 0;
            for (corpus) |s| total_len += s.len;
            if (total_len == 0) {
                return BpeTable{
                    .merges = std.mem.zeroes([MAX_MERGES]SymbolPair),
                    .num_merges = 0,
                };
            }

            var buf = try allocator.alloc(SymbolType, total_len);
            defer allocator.free(buf);
            {
                var off: usize = 0;
                for (corpus) |s| {
                    @memcpy(buf[off .. off + s.len], s);
                    off += s.len;
                }
            }

            // Track document boundaries to prevent merges across documents
            var boundaries = try allocator.alloc(usize, corpus.len);
            defer allocator.free(boundaries);
            {
                var off: usize = 0;
                for (corpus, 0..) |s, i| {
                    off += s.len;
                    boundaries[i] = off;
                }
            }

            var table = BpeTable{
                .merges = std.mem.zeroes([MAX_MERGES]SymbolPair),
                .num_merges = 0,
            };

            var len = total_len;

            for (0..MAX_MERGES) |merge_idx| {
                if (len < 2) break;

                const best = if (use_flat)
                    findBestPairFlat(buf[0..len], boundaries)
                else
                    try findBestPairMap(allocator, buf[0..len], boundaries);

                if (best.count < options.min_frequency) break; // below min frequency threshold

                const new_token: SymbolType = @intCast(token_offset_usize + merge_idx);
                table.merges[merge_idx] = .{ best.a, best.b };
                table.num_merges += 1;

                // Replace all occurrences of (best.a, best.b) with new_token.
                // Respect document boundaries.
                var write: usize = 0;
                var read: usize = 0;
                var bi: usize = 0;
                while (read < len) {
                    if (read + 1 < len) {
                        // Advance boundary pointer
                        while (bi < boundaries.len and boundaries[bi] <= read) {
                            bi += 1;
                        }
                        const crosses = (bi < boundaries.len and boundaries[bi] == read + 1);
                        if (!crosses and buf[read] == best.a and buf[read + 1] == best.b) {
                            buf[write] = new_token;
                            write += 1;
                            read += 2;
                            // Adjust subsequent boundaries
                            for (boundaries) |*bound| {
                                if (bound.* > read - 2) {
                                    if (bound.* > 0) bound.* -= 1;
                                }
                            }
                            continue;
                        }
                    }
                    buf[write] = buf[read];
                    write += 1;
                    read += 1;
                }
                len = write;
            }

            return table;
        }

        const BestPair = struct { a: SymbolType, b: SymbolType, count: u32 };

        /// Flat-array pair counting for small symbol types (≤ u8).
        fn findBestPairFlat(data: []const SymbolType, boundaries: []const usize) BestPair {
            var pair_counts: [symbol_count][symbol_count]u32 = std.mem.zeroes([symbol_count][symbol_count]u32);
            var bi: usize = 0;
            for (0..data.len - 1) |i| {
                while (bi < boundaries.len and boundaries[bi] <= i) {
                    bi += 1;
                }
                if (bi < boundaries.len and boundaries[bi] == i + 1) continue;
                pair_counts[data[i]][data[i + 1]] += 1;
            }

            var best = BestPair{ .a = 0, .b = 0, .count = 0 };
            for (0..symbol_count) |a| {
                for (0..symbol_count) |b| {
                    if (pair_counts[a][b] > best.count) {
                        best = .{
                            .a = @intCast(a),
                            .b = @intCast(b),
                            .count = pair_counts[a][b],
                        };
                    }
                }
            }
            return best;
        }

        /// HashMap-based pair counting for large symbol types (> u8).
        fn findBestPairMap(allocator: std.mem.Allocator, data: []const SymbolType, boundaries: []const usize) !BestPair {
            var pair_counts = PairMap.init(allocator);
            defer pair_counts.deinit();

            var bi: usize = 0;
            for (0..data.len - 1) |i| {
                while (bi < boundaries.len and boundaries[bi] <= i) {
                    bi += 1;
                }
                if (bi < boundaries.len and boundaries[bi] == i + 1) continue;
                const key = PairKey{ .a = data[i], .b = data[i + 1] };
                const gop = try pair_counts.getOrPut(key);
                if (gop.found_existing) {
                    gop.value_ptr.* += 1;
                } else {
                    gop.value_ptr.* = 1;
                }
            }

            var best = BestPair{ .a = 0, .b = 0, .count = 0 };
            var iter = pair_counts.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.* > best.count) {
                    best = .{
                        .a = entry.key_ptr.a,
                        .b = entry.key_ptr.b,
                        .count = entry.value_ptr.*,
                    };
                }
            }
            return best;
        }

        // ── Encode ──

        /// Encode using ArenaConfig. Working buffer uses `arena.temp`,
        /// final result allocated from `arena.output`.
        pub fn encodeArena(arena: ArenaConfig, table: *const BpeTable, input: []const SymbolType) ![]SymbolType {
            if (input.len == 0) {
                return try arena.output.alloc(SymbolType, 0);
            }

            var buf = try arena.temp.alloc(SymbolType, input.len);
            defer arena.temp.free(buf);
            @memcpy(buf, input);
            var len = input.len;

            for (0..table.num_merges) |m| {
                const pair = table.merges[m];
                const token: SymbolType = @intCast(token_offset_usize + m);

                var write_pos: usize = 0;
                var read_pos: usize = 0;
                while (read_pos < len) {
                    if (read_pos + 1 < len and buf[read_pos] == pair[0] and buf[read_pos + 1] == pair[1]) {
                        buf[write_pos] = token;
                        write_pos += 1;
                        read_pos += 2;
                    } else {
                        buf[write_pos] = buf[read_pos];
                        write_pos += 1;
                        read_pos += 1;
                    }
                }
                len = write_pos;
            }

            const result = try arena.output.alloc(SymbolType, len);
            @memcpy(result, buf[0..len]);
            return result;
        }

        /// Encode input symbols using the trained table.
        /// Applies merges in order (greedy from first merge to last).
        pub fn encode(allocator: std.mem.Allocator, table: *const BpeTable, input: []const SymbolType) ![]SymbolType {
            if (input.len == 0) {
                return try allocator.alloc(SymbolType, 0);
            }

            var buf = try allocator.alloc(SymbolType, input.len);
            @memcpy(buf, input);
            var len = input.len;

            for (0..table.num_merges) |m| {
                const pair = table.merges[m];
                const token: SymbolType = @intCast(token_offset_usize + m);

                var write_pos: usize = 0;
                var read_pos: usize = 0;
                while (read_pos < len) {
                    if (read_pos + 1 < len and buf[read_pos] == pair[0] and buf[read_pos + 1] == pair[1]) {
                        buf[write_pos] = token;
                        write_pos += 1;
                        read_pos += 2;
                    } else {
                        buf[write_pos] = buf[read_pos];
                        write_pos += 1;
                        read_pos += 1;
                    }
                }
                len = write_pos;
            }

            // Shrink to actual size
            if (len < buf.len) {
                const result = try allocator.realloc(buf, len);
                return result;
            }
            return buf;
        }

        // ── Serialization ──

        const symbol_size = @sizeOf(SymbolType);
        const merge_count_size = @sizeOf(MergeCountType);

        /// Serialize the merge table using ArenaConfig. Output from `arena.output`.
        pub fn serializeTableArena(arena: ArenaConfig, table: *const BpeTable) ![]u8 {
            return serializeTableImpl(arena.output, table);
        }

        /// Serialize the merge table into a byte slice.
        pub fn serializeTable(allocator: std.mem.Allocator, table: *const BpeTable) ![]u8 {
            return serializeTableImpl(allocator, table);
        }

        fn serializeTableImpl(allocator: std.mem.Allocator, table: *const BpeTable) ![]u8 {
            const nm: usize = table.num_merges;
            const size = merge_count_size + nm * 2 * symbol_size;
            var out = try allocator.alloc(u8, size);
            // Write num_merges
            @memcpy(out[0..merge_count_size], std.mem.asBytes(&table.num_merges));
            // Write merge pairs
            var off: usize = merge_count_size;
            for (0..nm) |i| {
                @memcpy(out[off .. off + symbol_size], std.mem.asBytes(&table.merges[i][0]));
                off += symbol_size;
                @memcpy(out[off .. off + symbol_size], std.mem.asBytes(&table.merges[i][1]));
                off += symbol_size;
            }
            return out;
        }

        /// Deserialize a table view from serialized data (no allocation).
        pub fn deserializeTableView(data: []const u8) !TableView {
            if (data.len < merge_count_size) return error.InvalidData;
            const num_merges = std.mem.bytesAsValue(MergeCountType, data[0..merge_count_size]).*;
            const needed = merge_count_size + @as(usize, num_merges) * 2 * symbol_size;
            if (data.len < needed) return error.InvalidData;

            var view = TableView{
                .merges = std.mem.zeroes([MAX_MERGES]SymbolPair),
                .num_merges = num_merges,
            };
            var off: usize = merge_count_size;
            for (0..num_merges) |i| {
                view.merges[i][0] = std.mem.bytesAsValue(SymbolType, data[off..][0..symbol_size]).*;
                off += symbol_size;
                view.merges[i][1] = std.mem.bytesAsValue(SymbolType, data[off..][0..symbol_size]).*;
                off += symbol_size;
            }
            return view;
        }

        // ── Decode ──

        /// Decode encoded symbols back to the original symbol stream.
        /// Uses a caller-provided buffer. Returns the slice of buf actually used.
        /// No allocation.
        pub fn decode(table: *const TableView, encoded: []const SymbolType, buf: []SymbolType) ![]SymbolType {
            var out_pos: usize = 0;

            for (encoded) |sym| {
                // Expand symbol (may be a merge token) into buf using a stack.
                // Stack depth bounded by max nesting = MAX_MERGES.
                var stack: [MAX_MERGES * 2]SymbolType = undefined;
                var sp: usize = 1;
                stack[0] = sym;

                while (sp > 0) {
                    sp -= 1;
                    const c = stack[sp];

                    if (c >= TOKEN_OFFSET and (@as(usize, c) - token_offset_usize) < table.num_merges) {
                        const merge_idx = @as(usize, c) - token_offset_usize;
                        // Push right first (so left is processed first)
                        if (sp + 2 > stack.len) return error.StackOverflow;
                        stack[sp] = table.merges[merge_idx][1];
                        sp += 1;
                        stack[sp] = table.merges[merge_idx][0];
                        sp += 1;
                    } else {
                        if (out_pos >= buf.len) return error.BufferTooSmall;
                        buf[out_pos] = c;
                        out_pos += 1;
                    }
                }
            }

            return buf[0..out_pos];
        }

        pub const Error = error{ InvalidData, StackOverflow, BufferTooSmall };
    };
}

// ── Tests ────────────────────────────────────────────────────
const testing = std.testing;

test "Bpe(u8, 128): empty corpus" {
    const B = Bpe(u8, 128);
    const corpus = [_][]const u8{};
    const table = try B.train(testing.allocator, &corpus, .{});
    try testing.expectEqual(@as(B.MergeCountType, 0), table.num_merges);
}

test "Bpe(u8, 128): single merge" {
    const B = Bpe(u8, 128);
    const corpus = [_][]const u8{"aaab"};
    const table = try B.train(testing.allocator, &corpus, .{});
    try testing.expectEqual(@as(B.MergeCountType, 1), table.num_merges);
    // Most frequent pair is "aa"
    try testing.expectEqual(B.SymbolPair{ 'a', 'a' }, table.merges[0]);
}

test "Bpe(u8, 128): encode/decode round-trip" {
    const B = Bpe(u8, 128);
    const corpus = [_][]const u8{ "abab", "abab", "abab" };
    const table = try B.train(testing.allocator, &corpus, .{});

    const input: []const u8 = "abab";
    const encoded = try B.encode(testing.allocator, &table, input);
    defer testing.allocator.free(encoded);

    // Encoded should be shorter
    try testing.expect(encoded.len <= input.len);

    // Decode
    const view = B.TableView{
        .merges = table.merges,
        .num_merges = table.num_merges,
    };
    var buf: [64]u8 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "Bpe(u8, 128): no worthwhile merges" {
    const B = Bpe(u8, 128);
    const corpus = [_][]const u8{"abcdefg"};
    const table = try B.train(testing.allocator, &corpus, .{});
    // All pairs appear only once → no merges (threshold is 2)
    try testing.expectEqual(@as(B.MergeCountType, 0), table.num_merges);
}

test "Bpe(u8, 128): serialization round-trip" {
    const B = Bpe(u8, 128);
    const corpus = [_][]const u8{ "ababab", "ababab" };
    const table = try B.train(testing.allocator, &corpus, .{});

    const data = try B.serializeTable(testing.allocator, &table);
    defer testing.allocator.free(data);

    const view = try B.deserializeTableView(data);
    try testing.expectEqual(table.num_merges, view.num_merges);
    for (0..table.num_merges) |i| {
        try testing.expectEqual(table.merges[i], view.merges[i]);
    }
}

test "Bpe(u8, 128): encode produces compression" {
    const B = Bpe(u8, 128);
    const corpus = [_][]const u8{ "the cat sat on the mat", "the cat sat on the mat" };
    const table = try B.train(testing.allocator, &corpus, .{});

    const input: []const u8 = "the cat sat on the mat";
    const encoded = try B.encode(testing.allocator, &table, input);
    defer testing.allocator.free(encoded);

    try testing.expect(encoded.len < input.len);

    // Verify round-trip
    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [256]u8 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "Bpe(u8, 128): multi-doc training respects boundaries" {
    const B = Bpe(u8, 128);
    // "xy" appears at boundary between docs but shouldn't be merged from split
    const corpus = [_][]const u8{ "aax", "yaa", "aax", "yaa" };
    const table = try B.train(testing.allocator, &corpus, .{});
    // Most frequent pair should be "aa" (appears 4 times), not "xy" (0 within-doc)
    try testing.expectEqual(B.SymbolPair{ 'a', 'a' }, table.merges[0]);
}

test "Bpe(u8, 128): decode with nested tokens" {
    const B = Bpe(u8, 128);
    // Train with enough data to create nested merges
    const corpus = [_][]const u8{ "abcabc", "abcabc", "abcabc", "abcabc" };
    const table = try B.train(testing.allocator, &corpus, .{});

    const input: []const u8 = "abcabc";
    const encoded = try B.encode(testing.allocator, &table, input);
    defer testing.allocator.free(encoded);

    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [256]u8 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualSlices(u8, input, decoded);
}

test "Bpe(u8, 128): token offset is 0x80" {
    // Classic BPE over bytes: original alphabet [0x00-0x7F], tokens [0x80-0xFF]
    const B = Bpe(u8, 128);
    try testing.expectEqual(@as(u8, 0x80), B.TOKEN_OFFSET);
    try testing.expectEqual(@as(usize, 128), B.MAX_MERGES);
    try testing.expectEqual(@as(usize, 128), B.ALPHABET_SIZE);
}

test "Bpe(u9, 256): full byte range + 256 merges" {
    // u9 = 512 values. 256 merges → TOKEN_OFFSET = 256.
    // Original alphabet [0, 256) covers all byte values.
    const B = Bpe(u9, 256);
    try testing.expectEqual(@as(u9, 256), B.TOKEN_OFFSET);
    try testing.expectEqual(@as(usize, 256), B.MAX_MERGES);
    try testing.expectEqual(@as(usize, 256), B.ALPHABET_SIZE);

    // Can train/encode byte-range data without losing high bytes
    const corpus_data = [_]u9{ 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xAB, 0xCD, 0xAB, 0xCD };
    const corpus = [_][]const u9{&corpus_data};
    const table = try B.train(testing.allocator, &corpus, .{});
    try testing.expect(table.num_merges > 0);
    // Most frequent pair should be (0x00, 0xFF) with 4 occurrences
    try testing.expectEqual(B.SymbolPair{ 0x00, 0xFF }, table.merges[0]);

    // Round-trip
    const input = [_]u9{ 0x00, 0xFF, 0xAB, 0xCD };
    const encoded = try B.encode(testing.allocator, &table, &input);
    defer testing.allocator.free(encoded);

    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [64]u9 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualSlices(u9, &input, decoded);
}

test "Bpe: min_frequency controls merge threshold" {
    // With min_frequency=5, pairs occurring <5 times should not be merged
    const B = Bpe(u8, 128);
    // "ab" appears 3 times, "cd" appears 3 times — neither reaches 5
    const corpus = [_][]const u8{ "ababab", "cdcdcd" };
    const table = try B.train(testing.allocator, &corpus, .{ .min_frequency = 5 });
    try testing.expectEqual(@as(B.MergeCountType, 0), table.num_merges);

    // Same corpus but with threshold 2 — should merge
    const table2 = try B.train(testing.allocator, &corpus, .{ .min_frequency = 2 });
    try testing.expect(table2.num_merges > 0);
}

test "Bpe(u6, 32): small alphabet BPE" {
    const B = Bpe(u6, 32);
    try testing.expectEqual(@as(u6, 32), B.TOKEN_OFFSET);
    try testing.expectEqual(@as(usize, 32), B.MAX_MERGES);

    // Train on u6-range data (values 0-31 for original alphabet)
    const corpus_data = [_]u6{ 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4 };
    const corpus = [_][]const u6{&corpus_data};
    const table = try B.train(testing.allocator, &corpus, .{});
    try testing.expect(table.num_merges > 0);
    // Most frequent pair should be (1, 2) with 4 occurrences
    try testing.expectEqual(B.SymbolPair{ 1, 2 }, table.merges[0]);

    // Round-trip
    const input = [_]u6{ 1, 2, 1, 2, 3, 4 };
    const encoded = try B.encode(testing.allocator, &table, &input);
    defer testing.allocator.free(encoded);
    try testing.expect(encoded.len <= input.len);

    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [64]u6 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualSlices(u6, &input, decoded);
}

test "Bpe(u16, 128): wide alphabet BPE with HashMap counting" {
    const B = Bpe(u16, 128);
    // TOKEN_OFFSET = 65536 - 128 = 65408. Full byte range fits in original alphabet.
    try testing.expectEqual(@as(u16, 65408), B.TOKEN_OFFSET);
    try testing.expectEqual(@as(usize, 128), B.MAX_MERGES);
    try testing.expectEqual(@as(usize, 65408), B.ALPHABET_SIZE);

    // Train on u16-range data
    const corpus_data = [_]u16{ 100, 200, 100, 200, 100, 200, 100, 200, 300, 400, 300, 400 };
    const corpus = [_][]const u16{&corpus_data};
    const table = try B.train(testing.allocator, &corpus, .{});
    try testing.expect(table.num_merges > 0);
    try testing.expectEqual(B.SymbolPair{ 100, 200 }, table.merges[0]);

    // Round-trip
    const input = [_]u16{ 100, 200, 100, 200, 300, 400 };
    const encoded = try B.encode(testing.allocator, &table, &input);
    defer testing.allocator.free(encoded);
    try testing.expect(encoded.len <= input.len);

    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [64]u16 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualSlices(u16, &input, decoded);
}

test "Bpe(u16, 128): serialization round-trip" {
    const B = Bpe(u16, 128);
    const corpus_data = [_]u16{ 10, 20, 10, 20, 10, 20, 10, 20 };
    const corpus = [_][]const u16{&corpus_data};
    const table = try B.train(testing.allocator, &corpus, .{});

    const data = try B.serializeTable(testing.allocator, &table);
    defer testing.allocator.free(data);

    const view = try B.deserializeTableView(data);
    try testing.expectEqual(table.num_merges, view.num_merges);
    for (0..table.num_merges) |i| {
        try testing.expectEqual(table.merges[i], view.merges[i]);
    }
}

test "Bpe(u4, 8): tiny alphabet" {
    const B = Bpe(u4, 8);
    // TOKEN_OFFSET = 16 - 8 = 8
    try testing.expectEqual(@as(u4, 8), B.TOKEN_OFFSET);
    try testing.expectEqual(@as(usize, 8), B.MAX_MERGES);
    try testing.expectEqual(@as(usize, 8), B.ALPHABET_SIZE);

    const corpus_data = [_]u4{ 1, 2, 1, 2, 1, 2, 3, 4, 3, 4 };
    const corpus = [_][]const u4{&corpus_data};
    const table = try B.train(testing.allocator, &corpus, .{});
    try testing.expect(table.num_merges > 0);

    // Round-trip
    const input = [_]u4{ 1, 2, 3, 4 };
    const encoded = try B.encode(testing.allocator, &table, &input);
    defer testing.allocator.free(encoded);

    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [32]u4 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualSlices(u4, &input, decoded);
}
