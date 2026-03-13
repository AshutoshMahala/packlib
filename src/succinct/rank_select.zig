//! Bit vector with O(1) rank and O(log n) select queries.
//!
//! Build-time: construct from bit data, serialize to `[]u8`.
//! Read-time: all queries on `[]const u8`, zero allocation.
//!
//! Rank: count of 1-bits (or 0-bits) in positions [0, pos).
//! Select: position of the i-th 1-bit (or 0-bit).
//!
//! Serialized format:
//!   [4/8 bytes: total_bits]
//!   [4/8 bytes: total_ones]
//!   [4/8 bytes: num_superblocks]
//!   [raw bit vector: ceil(total_bits / 8) bytes]
//!   [superblock table: num_superblocks × sizeof(IndexType)]
//!   [block table: num_blocks × u8 (delta within superblock)]
//!
//! Block structure: superblocks every 256 bits, blocks every 64 bits.
//! Rank = superblock cumulative count + block delta + popcount of remaining bits.

const std = @import("std");

/// Superblock covers this many bits.
const SUPERBLOCK_BITS: u32 = 256;
/// Block covers this many bits (must evenly divide SUPERBLOCK_BITS).
const BLOCK_BITS: u32 = 64;
/// Blocks per superblock.
const BLOCKS_PER_SUPER: u32 = SUPERBLOCK_BITS / BLOCK_BITS; // 4

pub fn RankSelect(comptime IndexType: type) type {
    comptime {
        const info = @typeInfo(IndexType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("RankSelect: IndexType must be an unsigned integer type, got " ++ @typeName(IndexType));
        if (@sizeOf(IndexType) < 2)
            @compileError("RankSelect: IndexType must be at least u16");
    }
    const index_size = @sizeOf(IndexType);

    return struct {
        const Self = @This();

        // ── Header offsets ──
        const HEADER_SIZE: u32 = index_size * 3; // total_bits, total_ones, num_superblocks

        // ── Build-time ──

        /// Build the serialized rank/select structure from raw bit data.
        /// Caller owns the returned slice.
        pub fn build(allocator: std.mem.Allocator, bits: []const u1) ![]u8 {
            const n: IndexType = @intCast(bits.len);
            const num_superblocks: IndexType = if (n == 0) 0 else @intCast((n - 1) / SUPERBLOCK_BITS + 1);
            const num_blocks: IndexType = if (n == 0) 0 else @intCast((n - 1) / BLOCK_BITS + 1);

            // Count total ones
            var total_ones: IndexType = 0;
            for (bits) |b| {
                total_ones += b;
            }

            // Pack raw bit vector
            const raw_bytes: u32 = @intCast((n + 7) / 8);

            // Calculate total size
            const superblock_table_size: u32 = @intCast(@as(u64, num_superblocks) * index_size);
            const block_table_size: u32 = @intCast(@as(u64, num_blocks));
            const total_size: u32 = HEADER_SIZE + raw_bytes + superblock_table_size + block_table_size;

            const data = try allocator.alloc(u8, total_size);
            @memset(data, 0);

            // Write header
            writeIndex(data, 0, n);
            writeIndex(data, index_size, total_ones);
            writeIndex(data, index_size * 2, num_superblocks);

            // Write raw bit vector
            const raw_start = HEADER_SIZE;
            for (0..@intCast(n)) |i| {
                if (bits[i] == 1) {
                    const byte_idx = raw_start + @as(u32, @intCast(i / 8));
                    const bit_idx: u3 = @intCast(7 - (i % 8)); // MSB-first
                    data[byte_idx] |= @as(u8, 1) << bit_idx;
                }
            }

            // Build superblock and block tables
            const super_start: u32 = raw_start + raw_bytes;
            const block_start: u32 = super_start + superblock_table_size;

            var cumulative: IndexType = 0;
            var block_idx: u32 = 0;

            for (0..@intCast(num_superblocks)) |si| {
                // Write cumulative count at superblock boundary
                writeIndex(data, super_start + @as(u32, @intCast(si)) * index_size, cumulative);

                const super_bit_start = si * SUPERBLOCK_BITS;
                var delta: u16 = 0;

                for (0..BLOCKS_PER_SUPER) |bi| {
                    const block_bit_start = super_bit_start + bi * BLOCK_BITS;
                    if (block_bit_start >= n) break;

                    // Write block delta (truncate to u8 — max written value is
                    // sum of ones from previous blocks within the superblock,
                    // which is at most SUPERBLOCK_BITS - BLOCK_BITS = 192)
                    data[block_start + block_idx] = @intCast(delta);
                    block_idx += 1;

                    // Count ones in this block
                    const block_bit_end = @min(block_bit_start + BLOCK_BITS, @as(usize, @intCast(n)));
                    var block_ones: u16 = 0;
                    for (block_bit_start..block_bit_end) |j| {
                        block_ones += bits[j];
                    }
                    delta += block_ones;
                    cumulative += @intCast(block_ones);
                }
            }

            return data;
        }

        // ── Read-time (all no-alloc on []const u8) ──

        /// Count of 1-bits in positions [0, pos).
        pub fn rank1(data: []const u8, bit_pos: IndexType) IndexType {
            const n = readTotalBits(data);
            if (bit_pos == 0) return 0;
            const clamped = @min(bit_pos, n);

            // When clamped falls exactly on a block boundary, we want the
            // *previous* block's complete count rather than indexing one past
            // the block table. Use (clamped - 1) to find the containing block,
            // then popcount up to the residual bits.
            const last_bit = clamped - 1;
            const superblock_idx = last_bit / SUPERBLOCK_BITS;
            const block_idx = last_bit / BLOCK_BITS;
            const bit_in_block = (last_bit % BLOCK_BITS) + 1; // bits to count within this block

            // Superblock cumulative count
            const super_start = superSectionStart(data);
            var count: IndexType = readIndex(data, super_start + @as(u32, @intCast(superblock_idx)) * index_size);

            // Block delta within superblock
            const blk_start = blockSectionStart(data);
            count += data[blk_start + @as(u32, @intCast(block_idx))];

            // Popcount of remaining bits within the block
            const block_bit_start: u32 = @intCast(@as(u64, block_idx) * BLOCK_BITS);
            const raw_start = HEADER_SIZE;
            count += @intCast(popcount(data[raw_start..], block_bit_start, @intCast(bit_in_block)));

            return count;
        }

        /// Count of 0-bits in positions [0, pos).
        pub fn rank0(data: []const u8, bit_pos: IndexType) IndexType {
            const n = readTotalBits(data);
            const clamped = @min(bit_pos, n);
            return clamped - rank1(data, clamped);
        }

        /// Position of the i-th 1-bit (0-indexed). Returns n if not found.
        pub fn select1(data: []const u8, i: IndexType) IndexType {
            const n = readTotalBits(data);
            const ones = readTotalOnes(data);
            if (i >= ones) return n;

            // Binary search: find smallest pos where rank1(pos) > i
            var lo: IndexType = 0;
            var hi: IndexType = n;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (rank1(data, mid + 1) <= i) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }

        /// Position of the i-th 0-bit (0-indexed). Returns n if not found.
        pub fn select0(data: []const u8, i: IndexType) IndexType {
            const n = readTotalBits(data);
            const zeros = n - readTotalOnes(data);
            if (i >= zeros) return n;

            // Binary search: find smallest pos where rank0(pos) > i
            var lo: IndexType = 0;
            var hi: IndexType = n;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (rank0(data, mid + 1) <= i) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }

        /// Read a single bit at position `bit_pos`.
        pub fn getBit(data: []const u8, bit_pos: IndexType) u1 {
            const raw_start = HEADER_SIZE;
            const byte_idx: u32 = raw_start + @as(u32, @intCast(bit_pos / 8));
            const bit_idx: u3 = @intCast(7 - @as(u32, @intCast(bit_pos % 8)));
            return @intCast((data[byte_idx] >> bit_idx) & 1);
        }

        /// Validate a serialized RankSelect blob. Call before querying
        /// if the data comes from an untrusted source.
        /// Returns `InvalidData` if the header or section layout is inconsistent.
        pub fn validate(data: []const u8) error{InvalidData}!void {
            if (data.len < HEADER_SIZE) return error.InvalidData;

            const total_bits = readTotalBits(data);
            const total_ones = readTotalOnes(data);
            const num_sb = readNumSuperblocks(data);

            if (total_ones > total_bits) return error.InvalidData;

            // Verify num_superblocks matches total_bits.
            const expected_sb: IndexType = if (total_bits == 0) 0 else @intCast((@as(u64, total_bits) - 1) / SUPERBLOCK_BITS + 1);
            if (num_sb != expected_sb) return error.InvalidData;

            // Verify derived section sizes fit within the blob.
            const raw_bytes: u64 = (@as(u64, total_bits) + 7) / 8;
            const super_table: u64 = @as(u64, num_sb) * index_size;
            const num_blocks: u64 = if (total_bits == 0) 0 else (@as(u64, total_bits) - 1) / BLOCK_BITS + 1;
            const expected_size: u64 = HEADER_SIZE + raw_bytes + super_table + num_blocks;
            if (expected_size > data.len) return error.InvalidData;
        }

        /// Total number of bits in the vector.
        pub fn totalBits(data: []const u8) IndexType {
            return readTotalBits(data);
        }

        /// Total number of 1-bits in the vector.
        pub fn totalOnes(data: []const u8) IndexType {
            return readTotalOnes(data);
        }

        // ── Internal helpers ──

        fn readTotalBits(data: []const u8) IndexType {
            return readIndex(data, 0);
        }

        fn readTotalOnes(data: []const u8) IndexType {
            return readIndex(data, index_size);
        }

        fn readNumSuperblocks(data: []const u8) IndexType {
            return readIndex(data, index_size * 2);
        }

        fn rawBytesCount(data: []const u8) u32 {
            const n = readTotalBits(data);
            return @intCast((@as(u64, n) + 7) / 8);
        }

        fn superSectionStart(data: []const u8) u32 {
            return HEADER_SIZE + rawBytesCount(data);
        }

        fn blockSectionStart(data: []const u8) u32 {
            const num_sb = readNumSuperblocks(data);
            return superSectionStart(data) + @as(u32, @intCast(num_sb)) * index_size;
        }

        fn readIndex(data: []const u8, offset: u32) IndexType {
            const bytes = data[offset..][0..index_size];
            return std.mem.readInt(IndexType, bytes, .little);
        }

        fn writeIndex(data: []u8, offset: u32, value: IndexType) void {
            const bytes = data[offset..][0..index_size];
            std.mem.writeInt(IndexType, bytes, value, .little);
        }

        /// Count set bits in `raw_bits` starting at `start_bit` for `count` bits.
        fn popcount(raw_bits: []const u8, start_bit: u32, bit_count: u32) u32 {
            var ones: u32 = 0;
            for (0..bit_count) |i| {
                const abs_bit = start_bit + @as(u32, @intCast(i));
                const byte_idx = abs_bit / 8;
                const bit_idx: u3 = @intCast(7 - (abs_bit % 8));
                ones += @intCast((raw_bits[byte_idx] >> bit_idx) & 1);
            }
            return ones;
        }
    };
}

// ── Tests ────────────────────────────────────────────────────
const testing = std.testing;
const RS = RankSelect(u32);

test "RankSelect: build and rank1 basic" {
    // Bit vector: 1,0,1,1,0,0,1,0
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    // rank1(0) = 0 (no bits before position 0)
    try testing.expectEqual(@as(u32, 0), RS.rank1(data, 0));
    // rank1(1) = 1 (bit 0 is 1)
    try testing.expectEqual(@as(u32, 1), RS.rank1(data, 1));
    // rank1(2) = 1 (bit 1 is 0)
    try testing.expectEqual(@as(u32, 1), RS.rank1(data, 2));
    // rank1(3) = 2
    try testing.expectEqual(@as(u32, 2), RS.rank1(data, 3));
    // rank1(4) = 3
    try testing.expectEqual(@as(u32, 3), RS.rank1(data, 4));
    // rank1(8) = 4 (total ones)
    try testing.expectEqual(@as(u32, 4), RS.rank1(data, 8));
}

test "RankSelect: rank0 + rank1 = pos" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    for (0..bits.len + 1) |i| {
        const idx: u32 = @intCast(i);
        try testing.expectEqual(idx, RS.rank0(data, idx) + RS.rank1(data, idx));
    }
}

test "RankSelect: rank1 matches naive scan" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    for (0..bits.len + 1) |pos| {
        var naive: u32 = 0;
        for (0..pos) |j| {
            naive += bits[j];
        }
        try testing.expectEqual(naive, RS.rank1(data, @intCast(pos)));
    }
}

test "RankSelect: select1 correctness" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    // 1-bits are at positions: 0, 2, 3, 6
    try testing.expectEqual(@as(u32, 0), RS.select1(data, 0));
    try testing.expectEqual(@as(u32, 2), RS.select1(data, 1));
    try testing.expectEqual(@as(u32, 3), RS.select1(data, 2));
    try testing.expectEqual(@as(u32, 6), RS.select1(data, 3));
}

test "RankSelect: select0 correctness" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    // 0-bits are at positions: 1, 4, 5, 7
    try testing.expectEqual(@as(u32, 1), RS.select0(data, 0));
    try testing.expectEqual(@as(u32, 4), RS.select0(data, 1));
    try testing.expectEqual(@as(u32, 5), RS.select0(data, 2));
    try testing.expectEqual(@as(u32, 7), RS.select0(data, 3));
}

test "RankSelect: select1(rank1(pos)) identity" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0, 1, 1 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    // For every 1-bit position, select1(rank1(pos+1)-1) == pos
    for (0..bits.len) |pos| {
        if (bits[pos] == 1) {
            const r = RS.rank1(data, @as(u32, @intCast(pos)) + 1);
            try testing.expectEqual(@as(u32, @intCast(pos)), RS.select1(data, r - 1));
        }
    }
}

test "RankSelect: rank1(select1(i)) == i" {
    const bits = [_]u1{ 0, 1, 0, 1, 1, 0, 1, 0, 0, 1 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    const ones = RS.totalOnes(data);
    for (0..ones) |i| {
        const s = RS.select1(data, @intCast(i));
        try testing.expectEqual(@as(u32, @intCast(i)), RS.rank1(data, s + 1) - 1);
    }
}

test "RankSelect: all zeros" {
    const bits = [_]u1{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    for (0..9) |i| {
        try testing.expectEqual(@as(u32, 0), RS.rank1(data, @intCast(i)));
    }
    try testing.expectEqual(@as(u32, 0), RS.select0(data, 0));
    try testing.expectEqual(@as(u32, 7), RS.select0(data, 7));
}

test "RankSelect: all ones" {
    const bits = [_]u1{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    for (0..9) |i| {
        try testing.expectEqual(@as(u32, @intCast(i)), RS.rank1(data, @intCast(i)));
        try testing.expectEqual(@as(u32, 0), RS.rank0(data, @intCast(i)));
    }
    try testing.expectEqual(@as(u32, 0), RS.select1(data, 0));
    try testing.expectEqual(@as(u32, 7), RS.select1(data, 7));
}

test "RankSelect: alternating bits" {
    // 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1 — crosses byte boundary
    var bits: [16]u1 = undefined;
    for (0..16) |i| {
        bits[i] = @intCast(i % 2);
    }
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    for (0..17) |i| {
        const idx: u32 = @intCast(i);
        try testing.expectEqual(idx, RS.rank0(data, idx) + RS.rank1(data, idx));
    }
    // rank1(16) = 8 (positions 1,3,5,7,9,11,13,15)
    try testing.expectEqual(@as(u32, 8), RS.rank1(data, 16));
}

test "RankSelect: getBit" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    for (0..bits.len) |i| {
        try testing.expectEqual(bits[i], RS.getBit(data, @intCast(i)));
    }
}

test "RankSelect: totalBits and totalOnes" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 12), RS.totalBits(data));
    try testing.expectEqual(@as(u32, 7), RS.totalOnes(data));
}

test "RankSelect: large vector (300+ bits) crosses superblock" {
    // 320 bits: alternating 1,0 pattern
    var bits: [320]u1 = undefined;
    for (0..320) |i| {
        bits[i] = @intCast(i % 2);
    }
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 320), RS.totalBits(data));
    try testing.expectEqual(@as(u32, 160), RS.totalOnes(data));

    // Verify rank at superblock boundary (256)
    try testing.expectEqual(@as(u32, 128), RS.rank1(data, 256));

    // Verify rank past superblock
    try testing.expectEqual(@as(u32, 160), RS.rank1(data, 320));

    // Spot-check select
    try testing.expectEqual(@as(u32, 1), RS.select1(data, 0)); // first 1-bit is at position 1
    try testing.expectEqual(@as(u32, 319), RS.select1(data, 159)); // last 1-bit
}

test "RankSelect: validate accepts valid blob" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0 };
    const data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);
    try RS.validate(data);
}

test "RankSelect: validate rejects too-short blob" {
    try testing.expectError(error.InvalidData, RS.validate(&[_]u8{ 0, 1 }));
    try testing.expectError(error.InvalidData, RS.validate(&[_]u8{}));
}

test "RankSelect: validate rejects inflated total_bits" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0 };
    var data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    // Corrupt total_bits to a huge value — derived section sizes will exceed blob.
    std.mem.writeInt(u32, data[0..4], 999999, .little);
    try testing.expectError(error.InvalidData, RS.validate(data));
}

test "RankSelect: validate rejects total_ones > total_bits" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0 };
    var data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    // Set total_ones to something larger than total_bits (8).
    std.mem.writeInt(u32, data[4..8], 100, .little);
    try testing.expectError(error.InvalidData, RS.validate(data));
}

test "RankSelect: validate rejects wrong num_superblocks" {
    const bits = [_]u1{ 1, 0, 1, 1, 0, 0, 1, 0 };
    var data = try RS.build(testing.allocator, &bits);
    defer testing.allocator.free(data);

    // Corrupt num_superblocks (should be 1 for 8 bits).
    std.mem.writeInt(u32, data[8..12], 5, .little);
    try testing.expectError(error.InvalidData, RS.validate(data));
}
