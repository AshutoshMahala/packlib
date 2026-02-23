//! Byte-pair encoding (BPE) — iterative digram replacement.
//!
//! Build-time: train a merge table from a corpus, encode byte streams.
//! Read-time: decode using serialized table + caller-provided buffer. Zero allocation.
//!
//! Assumes ASCII input (0x00–0x7F). BPE tokens use 0x80–0xFF (up to 128 merges).
//!
//! Serialized table format:
//!   [1 byte: num_merges]
//!   [num_merges * 2 bytes: pair (left, right) for token 0x80, 0x81, ...]

const std = @import("std");

pub fn Bpe(comptime IndexType: type) type {
    comptime {
        if (@sizeOf(IndexType) == 0) unreachable;
    }
    return struct {
        const Self = @This();

        pub const MAX_MERGES = 128;

        pub const BpeTable = struct {
            /// merges[i] = the pair (left, right) that maps to token 0x80+i.
            /// left and right can themselves be tokens from earlier merges.
            merges: [MAX_MERGES][2]u8,
            /// How many merges were trained.
            num_merges: u8,
        };

        /// Lightweight read-only view over serialized BPE table.
        pub const TableView = struct {
            merges: [MAX_MERGES][2]u8,
            num_merges: u8,
        };

        // ── Build-time ──

        /// Train a merge table from a corpus of byte slices.
        /// Iteratively finds the most frequent byte pair and replaces it with a new token.
        pub fn train(allocator: std.mem.Allocator, corpus: []const []const u8, max_merges: u16) !BpeTable {
            const capped: u8 = @intCast(@min(max_merges, MAX_MERGES));

            // Concatenate corpus into a mutable working buffer
            var total_len: usize = 0;
            for (corpus) |s| total_len += s.len;
            if (total_len == 0) {
                return BpeTable{
                    .merges = std.mem.zeroes([MAX_MERGES][2]u8),
                    .num_merges = 0,
                };
            }

            var buf = try allocator.alloc(u8, total_len);
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
                .merges = std.mem.zeroes([MAX_MERGES][2]u8),
                .num_merges = 0,
            };

            var len = total_len;

            for (0..capped) |merge_idx| {
                if (len < 2) break;

                // Count all digram frequencies (skip across boundaries)
                var pair_counts: [256][256]u32 = std.mem.zeroes([256][256]u32);
                var bi: usize = 0;
                for (0..len - 1) |i| {
                    // Advance boundary pointer
                    while (bi < boundaries.len and boundaries[bi] <= i) {
                        bi += 1;
                    }
                    // Skip if next byte crosses a boundary
                    if (bi < boundaries.len and boundaries[bi] == i + 1) continue;
                    pair_counts[buf[i]][buf[i + 1]] += 1;
                }

                // Find most frequent pair
                var best_a: u8 = 0;
                var best_b: u8 = 0;
                var best_count: u32 = 0;
                for (0..256) |a| {
                    for (0..256) |b| {
                        if (pair_counts[a][b] > best_count) {
                            best_count = pair_counts[a][b];
                            best_a = @intCast(a);
                            best_b = @intCast(b);
                        }
                    }
                }

                if (best_count < 2) break; // no pair worth merging

                const new_token: u8 = @intCast(0x80 + merge_idx);
                table.merges[merge_idx] = .{ best_a, best_b };
                table.num_merges += 1;

                // Replace all occurrences of (best_a, best_b) with new_token
                // Be careful to respect document boundaries
                var write: usize = 0;
                var read: usize = 0;
                bi = 0;
                while (read < len) {
                    if (read + 1 < len) {
                        // Check boundary
                        while (bi < boundaries.len and boundaries[bi] <= read) {
                            bi += 1;
                        }
                        const crosses = (bi < boundaries.len and boundaries[bi] == read + 1);
                        if (!crosses and buf[read] == best_a and buf[read + 1] == best_b) {
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

        /// Encode input bytes using the trained table.
        /// Applies merges in order (greedy from first merge to last).
        pub fn encode(allocator: std.mem.Allocator, table: *const BpeTable, input: []const u8) ![]u8 {
            if (input.len == 0) {
                return try allocator.alloc(u8, 0);
            }

            var buf = try allocator.alloc(u8, input.len);
            @memcpy(buf, input);
            var len = input.len;

            for (0..table.num_merges) |m| {
                const pair = table.merges[m];
                const token: u8 = @intCast(0x80 + m);

                var write: usize = 0;
                var read: usize = 0;
                while (read < len) {
                    if (read + 1 < len and buf[read] == pair[0] and buf[read + 1] == pair[1]) {
                        buf[write] = token;
                        write += 1;
                        read += 2;
                    } else {
                        buf[write] = buf[read];
                        write += 1;
                        read += 1;
                    }
                }
                len = write;
            }

            // Shrink to actual size
            if (len < buf.len) {
                const result = try allocator.realloc(buf, len);
                return result;
            }
            return buf;
        }

        /// Serialize the merge table into a byte slice.
        pub fn serializeTable(allocator: std.mem.Allocator, table: *const BpeTable) ![]u8 {
            const size = 1 + @as(usize, table.num_merges) * 2;
            var out = try allocator.alloc(u8, size);
            out[0] = table.num_merges;
            for (0..table.num_merges) |i| {
                out[1 + i * 2] = table.merges[i][0];
                out[1 + i * 2 + 1] = table.merges[i][1];
            }
            return out;
        }

        /// Deserialize a table view from serialized data (no allocation).
        pub fn deserializeTableView(data: []const u8) !TableView {
            if (data.len < 1) return error.InvalidData;
            const num_merges = data[0];
            if (data.len < 1 + @as(usize, num_merges) * 2) return error.InvalidData;

            var view = TableView{
                .merges = std.mem.zeroes([MAX_MERGES][2]u8),
                .num_merges = num_merges,
            };
            for (0..num_merges) |i| {
                view.merges[i][0] = data[1 + i * 2];
                view.merges[i][1] = data[1 + i * 2 + 1];
            }
            return view;
        }

        /// Decode encoded bytes back to the original byte stream.
        /// Uses a caller-provided buffer. Returns the slice of buf actually used.
        /// No allocation.
        pub fn decode(table: *const TableView, encoded: []const u8, buf: []u8) ![]u8 {
            // We need to expand tokens recursively. Use a stack.
            var out_pos: usize = 0;

            // Process each byte in the encoded sequence
            for (encoded) |b| {
                // Expand byte (may be a token) into buf
                var stack: [256]u8 = undefined;
                var sp: usize = 1;
                stack[0] = b;

                while (sp > 0) {
                    sp -= 1;
                    const c = stack[sp];

                    if (c >= 0x80 and (@as(usize, c) - 0x80) < table.num_merges) {
                        const merge_idx = @as(usize, c) - 0x80;
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

test "Bpe: empty corpus" {
    const B = Bpe(u32);
    const corpus = [_][]const u8{};
    const table = try B.train(testing.allocator, &corpus, 10);
    try testing.expectEqual(@as(u8, 0), table.num_merges);
}

test "Bpe: single merge" {
    const B = Bpe(u32);
    const corpus = [_][]const u8{"aaab"};
    const table = try B.train(testing.allocator, &corpus, 1);
    try testing.expectEqual(@as(u8, 1), table.num_merges);
    // Most frequent pair is "aa"
    try testing.expectEqual([2]u8{ 'a', 'a' }, table.merges[0]);
}

test "Bpe: encode/decode round-trip" {
    const B = Bpe(u32);
    const corpus = [_][]const u8{ "abab", "abab", "abab" };
    const table = try B.train(testing.allocator, &corpus, 4);

    const input = "abab";
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
    try testing.expectEqualStrings(input, decoded);
}

test "Bpe: no worthwhile merges" {
    const B = Bpe(u32);
    const corpus = [_][]const u8{"abcdefg"};
    const table = try B.train(testing.allocator, &corpus, 10);
    // All pairs appear only once → no merges (threshold is 2)
    try testing.expectEqual(@as(u8, 0), table.num_merges);
}

test "Bpe: serialization round-trip" {
    const B = Bpe(u32);
    const corpus = [_][]const u8{ "ababab", "ababab" };
    const table = try B.train(testing.allocator, &corpus, 3);

    const data = try B.serializeTable(testing.allocator, &table);
    defer testing.allocator.free(data);

    const view = try B.deserializeTableView(data);
    try testing.expectEqual(table.num_merges, view.num_merges);
    for (0..table.num_merges) |i| {
        try testing.expectEqual(table.merges[i], view.merges[i]);
    }
}

test "Bpe: encode produces compression" {
    const B = Bpe(u32);
    const corpus = [_][]const u8{ "the cat sat on the mat", "the cat sat on the mat" };
    const table = try B.train(testing.allocator, &corpus, 10);

    const input = "the cat sat on the mat";
    const encoded = try B.encode(testing.allocator, &table, input);
    defer testing.allocator.free(encoded);

    try testing.expect(encoded.len < input.len);

    // Verify round-trip
    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [256]u8 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualStrings(input, decoded);
}

test "Bpe: multi-doc training respects boundaries" {
    const B = Bpe(u32);
    // "xy" appears at boundary between docs but shouldn't be merged from split
    const corpus = [_][]const u8{ "aax", "yaa", "aax", "yaa" };
    const table = try B.train(testing.allocator, &corpus, 1);
    // Most frequent pair should be "aa" (appears 4 times), not "xy" (0 within-doc)
    try testing.expectEqual([2]u8{ 'a', 'a' }, table.merges[0]);
}

test "Bpe: decode with nested tokens" {
    const B = Bpe(u32);
    // Train with enough data to create nested merges
    const corpus = [_][]const u8{ "abcabc", "abcabc", "abcabc", "abcabc" };
    const table = try B.train(testing.allocator, &corpus, 3);

    const input = "abcabc";
    const encoded = try B.encode(testing.allocator, &table, input);
    defer testing.allocator.free(encoded);

    const view = B.TableView{ .merges = table.merges, .num_merges = table.num_merges };
    var buf: [256]u8 = undefined;
    const decoded = try B.decode(&view, encoded, &buf);
    try testing.expectEqualStrings(input, decoded);
}

test "Bpe: u16 index type" {
    const B = Bpe(u16);
    const corpus = [_][]const u8{ "hello hello", "hello hello" };
    const table = try B.train(testing.allocator, &corpus, 5);
    try testing.expect(table.num_merges > 0);
}
