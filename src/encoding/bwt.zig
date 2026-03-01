//! Burrows–Wheeler Transform (BWT) — reversible block-sorting transform.
//!
//! Rearranges input bytes so that characters from similar contexts are grouped
//! together, making the output highly compressible by entropy coders (e.g.,
//! after applying Move-to-Front + Huffman/ANS).
//!
//! **Build-time:** allocates (uses `ArenaConfig`).
//! **Decode-time:** allocates for the inverse transform.
//!
//! ## Algorithm
//!
//! Forward transform:
//! 1. Form all cyclic rotations of the input.
//! 2. Sort rotations lexicographically (suffix array construction).
//! 3. Output the last column + the row index of the original string.
//!
//! This implementation uses a simple suffix-array-based approach with
//! `std.sort` for inputs ≤ 64 KiB. For larger inputs, consider SA-IS.
//!
//! ## Serialized format
//!
//! ```text
//! [u32: original_index]  — row of the original string in sorted rotation matrix
//! [n bytes: bwt_output]  — last column of the sorted rotation matrix
//! ```
//!
//! ## References
//!
//! - Burrows & Wheeler (1994), "A Block-Sorting Lossless Data Compression Algorithm"
//! - Manzini (2001), "An analysis of the Burrows-Wheeler transform"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn Bwt(comptime IndexType: type) type {
    comptime {
        if (@sizeOf(IndexType) < 2) @compileError("IndexType must be at least u16");
    }
    const index_size = @sizeOf(IndexType);

    return struct {
        const Self = @This();

        pub const Error = error{
            EmptyInput,
            InputTooLarge,
            OutOfMemory,
            InvalidData,
        };

        /// Header: [u32: original_index]
        const HEADER_SIZE: u32 = @sizeOf(u32);

        // ── Forward Transform ──

        /// Compute the BWT of `input` using ArenaConfig.
        /// Suffix array and working memory use `arena.temp`.
        /// Output (header + BWT bytes) allocated from `arena.output`.
        pub fn forward(arena: ArenaConfig, input: []const u8) ![]u8 {
            return forwardImpl(arena.temp, arena.output, input);
        }

        /// Compute the BWT of `input` using a single allocator.
        pub fn forwardUniform(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
            return forwardImpl(allocator, allocator, input);
        }

        fn forwardImpl(
            temp: std.mem.Allocator,
            output: std.mem.Allocator,
            input: []const u8,
        ) ![]u8 {
            if (input.len == 0) return Error.EmptyInput;
            if (input.len > std.math.maxInt(IndexType)) return Error.InputTooLarge;

            const n: IndexType = @intCast(input.len);

            // Build suffix array (indices 0..n-1 sorted by cyclic rotation)
            const sa = try temp.alloc(IndexType, n);
            defer temp.free(sa);

            // Initialize with identity permutation
            for (0..@intCast(n)) |i| {
                sa[i] = @intCast(i);
            }

            // Sort by cyclic rotation using the input as context
            const Context = struct {
                data: []const u8,
                len: IndexType,
            };
            const ctx = Context{ .data = input, .len = n };

            std.sort.pdq(IndexType, sa, ctx, struct {
                fn lessThan(c: Context, a_idx: IndexType, b_idx: IndexType) bool {
                    // Compare cyclic rotations starting at a_idx and b_idx
                    const nn: usize = @intCast(c.len);
                    for (0..nn) |k| {
                        const ca = c.data[(@as(usize, a_idx) + k) % nn];
                        const cb = c.data[(@as(usize, b_idx) + k) % nn];
                        if (ca != cb) return ca < cb;
                    }
                    return false; // equal
                }
            }.lessThan);

            // Build output: header + last column
            const total_size = HEADER_SIZE + @as(u32, @intCast(n));
            const result = try output.alloc(u8, total_size);

            // Find original index and build last column
            var original_index: u32 = 0;
            for (0..@intCast(n)) |i| {
                if (sa[i] == 0) {
                    original_index = @intCast(i);
                }
                // Last column: character at (sa[i] + n - 1) % n
                const last_pos = (@as(usize, sa[i]) + @as(usize, @intCast(n)) - 1) % @as(usize, @intCast(n));
                result[HEADER_SIZE + i] = input[last_pos];
            }

            // Write header
            std.mem.writeInt(u32, result[0..4], original_index, .little);

            return result;
        }

        // ── Inverse Transform ──

        /// Reconstruct the original input from BWT data.
        /// Working memory uses `arena.temp`, result from `arena.output`.
        pub fn inverse(arena: ArenaConfig, data: []const u8) ![]u8 {
            return inverseImpl(arena.temp, arena.output, data);
        }

        /// Reconstruct the original input from BWT data using a single allocator.
        pub fn inverseUniform(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
            return inverseImpl(allocator, allocator, data);
        }

        fn inverseImpl(
            temp: std.mem.Allocator,
            output_alloc: std.mem.Allocator,
            data: []const u8,
        ) ![]u8 {
            if (data.len <= HEADER_SIZE) return Error.InvalidData;

            const original_index = std.mem.readInt(u32, data[0..4], .little);
            const bwt = data[HEADER_SIZE..];
            const n: u32 = @intCast(bwt.len);

            if (original_index >= n) return Error.InvalidData;

            // LF-mapping based inverse BWT (O(n) time, O(n) space)
            //
            // 1. Count character frequencies
            var counts: [256]u32 = std.mem.zeroes([256]u32);
            for (bwt) |c| counts[c] += 1;

            // 2. Compute cumulative counts (first occurrence of each char in F column)
            var cumul: [257]u32 = undefined;
            cumul[0] = 0;
            for (0..256) |i| {
                cumul[i + 1] = cumul[i] + counts[i];
            }

            // 3. Build LF-mapping: T[i] = cumul[bwt[i]] + rank of bwt[i] among bwt[0..i]
            const lf = try temp.alloc(u32, n);
            defer temp.free(lf);

            var occ: [256]u32 = std.mem.zeroes([256]u32);
            for (0..n) |i| {
                const c = bwt[i];
                lf[i] = cumul[c] + occ[c];
                occ[c] += 1;
            }

            // 4. Reconstruct original string by following LF-mapping backwards
            const result = try output_alloc.alloc(u8, n);
            var pos = original_index;
            var write_pos: u32 = n;
            while (write_pos > 0) {
                write_pos -= 1;
                result[write_pos] = bwt[pos];
                pos = lf[pos];
            }

            return result;
        }

        // ── Queries on serialized BWT data ──

        /// Get the original index from serialized BWT data.
        pub fn getOriginalIndex(data: []const u8) u32 {
            return std.mem.readInt(u32, data[0..4], .little);
        }

        /// Get the BWT output bytes (last column).
        pub fn getBwtBytes(data: []const u8) []const u8 {
            return data[HEADER_SIZE..];
        }

        /// Get the length of the original input.
        pub fn getLength(data: []const u8) u32 {
            return @intCast(data.len - HEADER_SIZE);
        }

        // ── Internal helpers ──

        fn readIdx(data: []const u8, offset: u32) IndexType {
            const bytes = data[offset..][0..index_size];
            return std.mem.readInt(IndexType, bytes, .little);
        }

        fn writeIdx(data: []u8, offset: u32, value: IndexType) void {
            const bytes = data[offset..][0..index_size];
            std.mem.writeInt(IndexType, bytes, value, .little);
        }
    };
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "BWT: basic round-trip" {
    const B = Bwt(u32);
    const input = "banana";
    const bwt_data = try B.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(bwt_data);

    const restored = try B.inverseUniform(testing.allocator, bwt_data);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, input, restored);
}

test "BWT: known BWT output for 'banana'" {
    const B = Bwt(u32);
    const bwt_data = try B.forwardUniform(testing.allocator, "banana");
    defer testing.allocator.free(bwt_data);

    const bwt_bytes = B.getBwtBytes(bwt_data);
    // Sorted rotations of "banana":
    // a$banan → last char: n
    // ana$ban → last char: n
    // anana$b → last char: b
    // banana$ → last char: a  (original, but cyclic so it's "banana" itself)
    // na$bana → last char: a
    // nana$ba → last char: a
    //
    // For cyclic BWT of "banana": sorted rotations:
    // "abanан" → "abanan", "ananab", "banana", "nabana", "nanaba", "nabanа"
    // The expected BWT is "nnbaaa"
    try testing.expectEqualSlices(u8, "nnbaaa", bwt_bytes);
}

test "BWT: single character" {
    const B = Bwt(u32);
    const bwt_data = try B.forwardUniform(testing.allocator, "x");
    defer testing.allocator.free(bwt_data);

    const restored = try B.inverseUniform(testing.allocator, bwt_data);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, "x", restored);
}

test "BWT: repeated character" {
    const B = Bwt(u32);
    const bwt_data = try B.forwardUniform(testing.allocator, "aaaa");
    defer testing.allocator.free(bwt_data);

    const bwt_bytes = B.getBwtBytes(bwt_data);
    try testing.expectEqualSlices(u8, "aaaa", bwt_bytes);

    const restored = try B.inverseUniform(testing.allocator, bwt_data);
    defer testing.allocator.free(restored);
    try testing.expectEqualSlices(u8, "aaaa", restored);
}

test "BWT: empty input returns error" {
    const B = Bwt(u32);
    try testing.expectError(error.EmptyInput, B.forwardUniform(testing.allocator, ""));
}

test "BWT: round-trip with all ASCII printable" {
    const B = Bwt(u32);
    var input: [95]u8 = undefined;
    for (0..95) |i| input[i] = @intCast(32 + i);

    const bwt_data = try B.forwardUniform(testing.allocator, &input);
    defer testing.allocator.free(bwt_data);

    const restored = try B.inverseUniform(testing.allocator, bwt_data);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, &input, restored);
}

test "BWT: round-trip with binary data" {
    const B = Bwt(u32);
    var input: [256]u8 = undefined;
    for (0..256) |i| input[i] = @intCast(i);

    const bwt_data = try B.forwardUniform(testing.allocator, &input);
    defer testing.allocator.free(bwt_data);

    const restored = try B.inverseUniform(testing.allocator, bwt_data);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, &input, restored);
}

test "BWT: round-trip with repetitive text" {
    const B = Bwt(u32);
    const input = "abcabcabcabcabcabc";
    const bwt_data = try B.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(bwt_data);

    const restored = try B.inverseUniform(testing.allocator, bwt_data);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, input, restored);
}

test "BWT: BWT clusters similar contexts" {
    const B = Bwt(u32);
    const input = "the_cat_sat_on_the_mat";
    const bwt_data = try B.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(bwt_data);

    // The BWT should cluster characters from similar contexts.
    // Not testing exact output but verifying round-trip works.
    const restored = try B.inverseUniform(testing.allocator, bwt_data);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, input, restored);
}

test "BWT: original index is valid" {
    const B = Bwt(u32);
    const input = "hello world";
    const bwt_data = try B.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(bwt_data);

    const idx = B.getOriginalIndex(bwt_data);
    try testing.expect(idx < @as(u32, @intCast(input.len)));

    const len = B.getLength(bwt_data);
    try testing.expectEqual(@as(u32, @intCast(input.len)), len);
}

test "BWT: ArenaConfig variant" {
    const B = Bwt(u32);
    const arena = ArenaConfig.uniform(testing.allocator);
    const bwt_data = try B.forward(arena, "test");
    defer testing.allocator.free(bwt_data);

    const restored = try B.inverse(arena, bwt_data);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, "test", restored);
}

test "BWT: u16 index type" {
    const B = Bwt(u16);
    const input = "abracadabra";
    const bwt_data = try B.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(bwt_data);

    const restored = try B.inverseUniform(testing.allocator, bwt_data);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, input, restored);
}

test "BWT: inverse detects invalid data" {
    const B = Bwt(u32);
    // Data too short (just header, no BWT bytes)
    const short = [_]u8{ 0, 0, 0, 0 };
    try testing.expectError(error.InvalidData, B.inverseUniform(testing.allocator, &short));

    // Invalid original index (>= n)
    var bad_idx: [5]u8 = undefined;
    std.mem.writeInt(u32, bad_idx[0..4], 99, .little); // index=99 but n=1
    bad_idx[4] = 'a';
    try testing.expectError(error.InvalidData, B.inverseUniform(testing.allocator, &bad_idx));
}
