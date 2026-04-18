//! Front Coding — prefix-compressed sorted string sequences.
//!
//! Compresses a sorted list of strings by storing each string as a
//! (shared_prefix_length, suffix) pair relative to its predecessor.
//! Achieves excellent compression for sorted dictionaries, URL lists,
//! and other lexicographically ordered string collections.
//!
//! Uses bucket-based layout: every `bucket_size` strings, the full string
//! is stored (for random access). Within a bucket, each subsequent string
//! stores only the prefix share length + remaining suffix.
//!
//! **Build-time:** allocates (uses `ArenaConfig`).
//! **Read-time:** needs a temp buffer for reconstruction, otherwise zero alloc.
//!
//! ## Serialized format
//!
//! ```text
//! [IndexType: num_strings]
//! [u16: bucket_size]
//! [IndexType: num_buckets]
//! [num_buckets × IndexType: byte offset of each bucket in the data section]
//! [variable: bucket data]
//! ```
//!
//! String lengths within each bucket are always stored as `u16`.
//! `IndexType` controls the count and offset fields in the header; it must be
//! wide enough to address the total serialized size (e.g. `u16` for < 64 KB,
//! `u32` for < 4 GB).
//!
//! Each bucket:
//! ```text
//! [u16: full_string_length] [bytes: full_string]
//! For each subsequent string in bucket:
//!   [u16: shared_prefix_length] [u16: suffix_length] [bytes: suffix]
//! ```
//!
//! ## References
//!
//! - Witten, Moffat, Bell (1999), "Managing Gigabytes"
//! - Bender et al. (2006), "Front Coding for Space-Efficient Text Retrieval"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn FrontCoding(comptime IndexType: type) type {
    comptime {
        const info = @typeInfo(IndexType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("FrontCoding: IndexType must be an unsigned integer type, got " ++ @typeName(IndexType));
        if (@bitSizeOf(IndexType) < 16)
            @compileError("FrontCoding: IndexType must be at least u16");
        if (@bitSizeOf(IndexType) > 64)
            @compileError("FrontCoding: IndexType must be at most u64");
    }

    const index_size = @sizeOf(IndexType);

    return struct {
        pub const Error = error{
            EmptyInput,
            NotSorted,
            StringTooLong,
            InvalidData,
            BufferTooSmall,
            OutOfMemory,
        };

        const HEADER_SIZE: usize = index_size + 2 + index_size; // num_strings + bucket_size + num_buckets

        /// Default bucket size (strings per bucket).
        pub const DEFAULT_BUCKET_SIZE: u16 = 16;

        // ── Build-time ──

        /// Encode a sorted list of strings using ArenaConfig.
        /// Temp buffers use `arena.temp`, final output from `arena.output`.
        pub fn encode(arena: ArenaConfig, strings: []const []const u8) ![]u8 {
            return encodeWithBucketSize(arena, strings, DEFAULT_BUCKET_SIZE);
        }

        /// Encode with a specified bucket size.
        pub fn encodeWithBucketSize(arena: ArenaConfig, strings: []const []const u8, bucket_size: u16) ![]u8 {
            return encodeImpl(arena.temp, arena.output, strings, bucket_size);
        }

        /// Encode with a single allocator.
        pub fn encodeUniform(allocator: std.mem.Allocator, strings: []const []const u8) ![]u8 {
            return encodeImpl(allocator, allocator, strings, DEFAULT_BUCKET_SIZE);
        }

        fn encodeImpl(
            temp: std.mem.Allocator,
            output: std.mem.Allocator,
            strings: []const []const u8,
            bucket_size: u16,
        ) ![]u8 {
            if (strings.len == 0) return Error.EmptyInput;
            if (strings.len > std.math.maxInt(IndexType)) return Error.InvalidData;
            const n: IndexType = @intCast(strings.len);

            // Verify sorted order
            for (1..strings.len) |i| {
                if (std.mem.order(u8, strings[i], strings[i - 1]) == .lt) {
                    return Error.NotSorted;
                }
            }

            // Verify no string exceeds u16 max
            for (strings) |s| {
                if (s.len > std.math.maxInt(u16)) return Error.StringTooLong;
            }

            const num_buckets: IndexType = @intCast((n + bucket_size - 1) / bucket_size);

            // Phase 1: Encode all buckets into a temporary buffer
            var data_buf = std.ArrayListUnmanaged(u8){};
            defer data_buf.deinit(temp);

            // Store raw byte offsets as usize during construction; cast to IndexType when serializing.
            var bucket_offsets = try temp.alloc(usize, num_buckets);
            defer temp.free(bucket_offsets);

            for (0..num_buckets) |bi| {
                bucket_offsets[bi] = data_buf.items.len;

                const bucket_start: IndexType = @intCast(bi * bucket_size);
                const bucket_end: IndexType = @min(bucket_start + bucket_size, n);

                // First string in bucket: store in full
                const first = strings[bucket_start];
                const first_len: u16 = @intCast(first.len);
                try appendU16(&data_buf, temp, first_len);
                try data_buf.appendSlice(temp, first);

                // Subsequent strings: prefix share + suffix
                for (bucket_start + 1..bucket_end) |i| {
                    const prev = strings[i - 1];
                    const curr = strings[i];

                    const shared = commonPrefixLen(prev, curr);
                    const suffix_len: u16 = @intCast(curr.len - shared);

                    try appendU16(&data_buf, temp, @intCast(shared));
                    try appendU16(&data_buf, temp, suffix_len);
                    try data_buf.appendSlice(temp, curr[shared..]);
                }
            }

            // Phase 2: Assemble output
            const offsets_size: usize = @as(usize, num_buckets) * index_size;
            const total_size: usize = HEADER_SIZE + offsets_size + data_buf.items.len;
            const result = try output.alloc(u8, total_size);

            // Header
            std.mem.writeInt(IndexType, result[0..index_size], n, .little);
            std.mem.writeInt(u16, result[index_size..][0..2], bucket_size, .little);
            std.mem.writeInt(IndexType, result[index_size + 2 ..][0..index_size], num_buckets, .little);

            // Bucket offsets
            for (0..num_buckets) |bi| {
                const abs_off: usize = HEADER_SIZE + offsets_size + bucket_offsets[bi];
                if (abs_off > std.math.maxInt(IndexType)) return Error.InvalidData;
                const slot: usize = HEADER_SIZE + bi * index_size;
                std.mem.writeInt(IndexType, result[slot..][0..index_size], @intCast(abs_off), .little);
            }

            // Bucket data
            const data_start = HEADER_SIZE + offsets_size;
            @memcpy(result[data_start..][0..data_buf.items.len], data_buf.items);

            return result;
        }

        // ── Read-time queries ──

        /// Number of encoded strings.
        pub fn count(data: []const u8) IndexType {
            return std.mem.readInt(IndexType, data[0..index_size], .little);
        }

        /// Get the i-th string, writing it into `buf`.
        /// Returns the slice of `buf` actually used.
        pub fn get(data: []const u8, index: IndexType, buf: []u8) ![]u8 {
            const n = std.mem.readInt(IndexType, data[0..index_size], .little);
            const bucket_size = std.mem.readInt(u16, data[index_size..][0..2], .little);

            if (index >= n) return Error.InvalidData;

            const bucket_idx: IndexType = index / @as(IndexType, @intCast(bucket_size));
            const idx_in_bucket: IndexType = index % @as(IndexType, @intCast(bucket_size));

            // Read bucket offset
            const off_pos: usize = HEADER_SIZE + @as(usize, bucket_idx) * index_size;
            const bucket_offset = std.mem.readInt(IndexType, data[off_pos..][0..index_size], .little);

            var pos: usize = bucket_offset;

            // Read the first string in the bucket (full string)
            const first_len = readU16(data, pos);
            pos += 2;
            if (first_len > buf.len) return Error.BufferTooSmall;
            @memcpy(buf[0..first_len], data[pos..][0..first_len]);
            pos += first_len;
            var current_len: u16 = first_len;

            // Walk through subsequent strings in the bucket
            for (0..idx_in_bucket) |_| {
                const shared: u16 = readU16(data, pos);
                pos += 2;
                const suffix_len: u16 = readU16(data, pos);
                pos += 2;
                const new_len = shared + suffix_len;
                if (new_len > buf.len) return Error.BufferTooSmall;

                // The shared prefix is already in buf[0..shared].
                // Copy the suffix after it.
                @memcpy(buf[shared..][0..suffix_len], data[pos..][0..suffix_len]);
                pos += suffix_len;
                current_len = new_len;
            }

            return buf[0..current_len];
        }

        /// Search for a string using binary search on bucket headers.
        /// Returns the index if found, or null.
        pub fn find(data: []const u8, target: []const u8, buf: []u8) !?IndexType {
            const n = count(data);
            if (n == 0) return null;

            // Binary search
            var lo: IndexType = 0;
            var hi: IndexType = n;
            while (lo < hi) {
                const mid: IndexType = lo + (hi - lo) / 2;
                const s = try get(data, mid, buf);
                switch (std.mem.order(u8, s, target)) {
                    .lt => lo = mid + 1,
                    .gt => hi = mid,
                    .eq => return mid,
                }
            }
            return null;
        }

        // ── Helpers ──

        fn commonPrefixLen(a: []const u8, b: []const u8) usize {
            const max = @min(a.len, b.len);
            for (0..max) |i| {
                if (a[i] != b[i]) return i;
            }
            return max;
        }

        fn readU16(data: []const u8, offset: usize) u16 {
            return std.mem.readInt(u16, data[offset..][0..2], .little);
        }

        fn appendU16(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: u16) !void {
            var buf: [2]u8 = undefined;
            std.mem.writeInt(u16, &buf, value, .little);
            try list.appendSlice(allocator, &buf);
        }
    }; // return struct
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;
const DefaultFC = FrontCoding(u32);
test "FrontCoding: basic round-trip" {
    const strings = [_][]const u8{ "apple", "application", "apply", "banana", "bandana" };
    const data = try DefaultFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 5), DefaultFC.count(data));

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try DefaultFC.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, expected, got);
    }
}

test "FrontCoding: single string" {
    const strings = [_][]const u8{"hello"};
    const data = try DefaultFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 1), DefaultFC.count(data));

    var buf: [256]u8 = undefined;
    const got = try DefaultFC.get(data, 0, &buf);
    try testing.expectEqualSlices(u8, "hello", got);
}

test "FrontCoding: identical strings" {
    const strings = [_][]const u8{ "abc", "abc", "abc" };
    const data = try DefaultFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    for (0..3) |i| {
        const got = try DefaultFC.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, "abc", got);
    }
}

test "FrontCoding: no shared prefix" {
    const strings = [_][]const u8{ "aaa", "bbb", "ccc" };
    const data = try DefaultFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try DefaultFC.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, expected, got);
    }
}

test "FrontCoding: long shared prefixes" {
    const strings = [_][]const u8{
        "http://www.example.com/page1",
        "http://www.example.com/page2",
        "http://www.example.com/page3",
        "http://www.example.com/page99",
    };
    const data = try DefaultFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try DefaultFC.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, expected, got);
    }

    // Space savings: full strings = 110 bytes, front-coded should be much less
    // Header + offsets + data should be < 80% of raw size
    const raw_size: usize = blk: {
        var s: usize = 0;
        for (strings) |str| s += str.len;
        break :blk s;
    };
    try testing.expect(data.len < raw_size);
}

test "FrontCoding: binary search (find)" {
    const strings = [_][]const u8{ "alpha", "beta", "gamma", "delta" };

    // Must be sorted for binary search to work!
    // Note: "alpha" < "beta" < "delta" < "gamma"
    const sorted = [_][]const u8{ "alpha", "beta", "delta", "gamma" };
    const data = try DefaultFC.encodeUniform(testing.allocator, &sorted);
    defer testing.allocator.free(data);
    _ = strings;

    var buf: [256]u8 = undefined;

    try testing.expectEqual(@as(?u32, 0), try DefaultFC.find(data, "alpha", &buf));
    try testing.expectEqual(@as(?u32, 1), try DefaultFC.find(data, "beta", &buf));
    try testing.expectEqual(@as(?u32, 2), try DefaultFC.find(data, "delta", &buf));
    try testing.expectEqual(@as(?u32, 3), try DefaultFC.find(data, "gamma", &buf));
    try testing.expectEqual(@as(?u32, null), try DefaultFC.find(data, "omega", &buf));
    try testing.expectEqual(@as(?u32, null), try DefaultFC.find(data, "aaa", &buf));
}

test "FrontCoding: bucket boundary" {
    // Create enough strings to span multiple buckets (bucket_size=4)
    const strings = [_][]const u8{
        "a01", "a02", "a03", "a04", // bucket 0
        "a05", "a06", "a07", "a08", // bucket 1
        "a09", "a10",
    };
    const arena = ArenaConfig.uniform(testing.allocator);
    const data = try DefaultFC.encodeWithBucketSize(arena, &strings, 4);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try DefaultFC.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, expected, got);
    }
}

test "FrontCoding: empty strings" {
    const strings = [_][]const u8{ "", "", "a", "b" };
    const data = try DefaultFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    const got0 = try DefaultFC.get(data, 0, &buf);
    try testing.expectEqual(@as(usize, 0), got0.len);

    const got2 = try DefaultFC.get(data, 2, &buf);
    try testing.expectEqualSlices(u8, "a", got2);
}

test "FrontCoding: not sorted returns error" {
    const strings = [_][]const u8{ "banana", "apple" };
    try testing.expectError(error.NotSorted, DefaultFC.encodeUniform(testing.allocator, &strings));
}

test "FrontCoding: empty input returns error" {
    const empty: []const []const u8 = &.{};
    try testing.expectError(error.EmptyInput, DefaultFC.encodeUniform(testing.allocator, empty));
}

// ── u16 IndexType tests ──

test "FrontCoding: u16 IndexType round-trip" {
    const SmallFC = FrontCoding(u16);
    const strings = [_][]const u8{ "cat", "cats", "dog", "dogs", "fish" };
    const data = try SmallFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u16, 5), SmallFC.count(data));

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try SmallFC.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, expected, got);
    }
}

test "FrontCoding: u16 IndexType is smaller than u32" {
    const SmallFC = FrontCoding(u16);
    const strings = [_][]const u8{ "apple", "application", "apply", "banana", "bandana" };
    const small = try SmallFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(small);
    const large = try DefaultFC.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(large);

    // u16 header fields take less space than u32
    try testing.expect(small.len < large.len);
}
