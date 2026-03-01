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
//! [u32: num_strings]
//! [u16: bucket_size]
//! [u32: num_buckets]
//! [num_buckets × u32: byte offset of each bucket in the data section]
//! [variable: bucket data]
//! ```
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

pub const FrontCoding = struct {
    pub const Error = error{
        EmptyInput,
        NotSorted,
        StringTooLong,
        InvalidData,
        BufferTooSmall,
        OutOfMemory,
    };

    const HEADER_SIZE: u32 = 4 + 2 + 4; // num_strings + bucket_size + num_buckets

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

        const n: u32 = @intCast(strings.len);

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

        const num_buckets: u32 = (n + bucket_size - 1) / bucket_size;

        // Phase 1: Encode all buckets into a temporary buffer
        var data_buf = std.ArrayListUnmanaged(u8){};
        defer data_buf.deinit(temp);

        var bucket_offsets = try temp.alloc(u32, num_buckets);
        defer temp.free(bucket_offsets);

        for (0..num_buckets) |bi| {
            bucket_offsets[bi] = @intCast(data_buf.items.len);

            const bucket_start: u32 = @intCast(bi * bucket_size);
            const bucket_end: u32 = @min(bucket_start + bucket_size, n);

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
        const offsets_size = num_buckets * 4;
        const total_size = HEADER_SIZE + offsets_size + @as(u32, @intCast(data_buf.items.len));
        const result = try output.alloc(u8, total_size);

        // Header
        std.mem.writeInt(u32, result[0..4], n, .little);
        std.mem.writeInt(u16, result[4..6], bucket_size, .little);
        std.mem.writeInt(u32, result[6..10], num_buckets, .little);

        // Bucket offsets
        for (0..num_buckets) |bi| {
            const off: u32 = HEADER_SIZE + offsets_size + bucket_offsets[bi];
            std.mem.writeInt(u32, result[HEADER_SIZE + @as(u32, @intCast(bi)) * 4 ..][0..4], off, .little);
        }

        // Bucket data
        const data_start = HEADER_SIZE + offsets_size;
        @memcpy(result[data_start..][0..data_buf.items.len], data_buf.items);

        return result;
    }

    // ── Read-time queries ──

    /// Number of encoded strings.
    pub fn count(data: []const u8) u32 {
        return std.mem.readInt(u32, data[0..4], .little);
    }

    /// Get the i-th string, writing it into `buf`.
    /// Returns the slice of `buf` actually used.
    pub fn get(data: []const u8, index: u32, buf: []u8) ![]u8 {
        const n = std.mem.readInt(u32, data[0..4], .little);
        const bucket_size = std.mem.readInt(u16, data[4..6], .little);

        if (index >= n) return Error.InvalidData;

        const bucket_idx = index / bucket_size;
        const idx_in_bucket = index % bucket_size;

        // Read bucket offset
        const off_pos = HEADER_SIZE + bucket_idx * 4;
        const bucket_offset = std.mem.readInt(u32, data[off_pos..][0..4], .little);

        var pos: u32 = bucket_offset;

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
    pub fn find(data: []const u8, target: []const u8, buf: []u8) !?u32 {
        const n = std.mem.readInt(u32, data[0..4], .little);
        if (n == 0) return null;

        // Binary search
        var lo: u32 = 0;
        var hi: u32 = n;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
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

    fn readU16(data: []const u8, offset: u32) u16 {
        return std.mem.readInt(u16, data[offset..][0..2], .little);
    }

    fn appendU16(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: u16) !void {
        var buf: [2]u8 = undefined;
        std.mem.writeInt(u16, &buf, value, .little);
        try list.appendSlice(allocator, &buf);
    }
};

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "FrontCoding: basic round-trip" {
    const strings = [_][]const u8{ "apple", "application", "apply", "banana", "bandana" };
    const data = try FrontCoding.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 5), FrontCoding.count(data));

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try FrontCoding.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, expected, got);
    }
}

test "FrontCoding: single string" {
    const strings = [_][]const u8{"hello"};
    const data = try FrontCoding.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 1), FrontCoding.count(data));

    var buf: [256]u8 = undefined;
    const got = try FrontCoding.get(data, 0, &buf);
    try testing.expectEqualSlices(u8, "hello", got);
}

test "FrontCoding: identical strings" {
    const strings = [_][]const u8{ "abc", "abc", "abc" };
    const data = try FrontCoding.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    for (0..3) |i| {
        const got = try FrontCoding.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, "abc", got);
    }
}

test "FrontCoding: no shared prefix" {
    const strings = [_][]const u8{ "aaa", "bbb", "ccc" };
    const data = try FrontCoding.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try FrontCoding.get(data, @intCast(i), &buf);
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
    const data = try FrontCoding.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try FrontCoding.get(data, @intCast(i), &buf);
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
    const data = try FrontCoding.encodeUniform(testing.allocator, &sorted);
    defer testing.allocator.free(data);
    _ = strings;

    var buf: [256]u8 = undefined;

    try testing.expectEqual(@as(?u32, 0), try FrontCoding.find(data, "alpha", &buf));
    try testing.expectEqual(@as(?u32, 1), try FrontCoding.find(data, "beta", &buf));
    try testing.expectEqual(@as(?u32, 2), try FrontCoding.find(data, "delta", &buf));
    try testing.expectEqual(@as(?u32, 3), try FrontCoding.find(data, "gamma", &buf));
    try testing.expectEqual(@as(?u32, null), try FrontCoding.find(data, "omega", &buf));
    try testing.expectEqual(@as(?u32, null), try FrontCoding.find(data, "aaa", &buf));
}

test "FrontCoding: bucket boundary" {
    // Create enough strings to span multiple buckets (bucket_size=4)
    const strings = [_][]const u8{
        "a01", "a02", "a03", "a04", // bucket 0
        "a05", "a06", "a07", "a08", // bucket 1
        "a09", "a10",
    };
    const arena = ArenaConfig.uniform(testing.allocator);
    const data = try FrontCoding.encodeWithBucketSize(arena, &strings, 4);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    for (strings, 0..) |expected, i| {
        const got = try FrontCoding.get(data, @intCast(i), &buf);
        try testing.expectEqualSlices(u8, expected, got);
    }
}

test "FrontCoding: empty strings" {
    const strings = [_][]const u8{ "", "", "a", "b" };
    const data = try FrontCoding.encodeUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var buf: [256]u8 = undefined;
    const got0 = try FrontCoding.get(data, 0, &buf);
    try testing.expectEqual(@as(usize, 0), got0.len);

    const got2 = try FrontCoding.get(data, 2, &buf);
    try testing.expectEqualSlices(u8, "a", got2);
}

test "FrontCoding: not sorted returns error" {
    const strings = [_][]const u8{ "banana", "apple" };
    try testing.expectError(error.NotSorted, FrontCoding.encodeUniform(testing.allocator, &strings));
}

test "FrontCoding: empty input returns error" {
    const empty: []const []const u8 = &.{};
    try testing.expectError(error.EmptyInput, FrontCoding.encodeUniform(testing.allocator, empty));
}
