//! Delta encoding + variable-length integer encoding for sorted integer sequences.
//!
//! Build-time: encode a monotone sequence into a compact byte representation.
//! Read-time: random-access `get(index)` without decoding the entire sequence.
//!
//! Uses block-based layout with absolute values at block boundaries for random access.
//! LEB128 varint encoding for delta values.
//!
//! Serialized format:
//!   [4 bytes: count (u32)]
//!   [2 bytes: block_size (u16)]
//!   [sizeof(ValueType) bytes per block: absolute values at block starts]
//!   [4 bytes per block: byte offset into delta section for each block]
//!   [variable: LEB128-encoded deltas for each block]

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn DeltaVarint(comptime ValueType: type) type {
    const value_size = @sizeOf(ValueType);

    return struct {
        const Self = @This();

        // Header: count(u32) + block_size(u16)
        const HEADER_SIZE: u32 = 6;

        /// Encode a sorted sequence using ArenaConfig with default block size (64).
        /// Temp buffers use `arena.temp`, output uses `arena.output`.
        pub fn encodeArena(arena: ArenaConfig, values: []const ValueType) ![]u8 {
            return encodeWithBlockSizeArena(arena, values, 64);
        }

        /// Encode a sorted sequence of values with default block size (64).
        pub fn encode(allocator: std.mem.Allocator, values: []const ValueType) ![]u8 {
            return encodeWithBlockSize(allocator, values, 64);
        }

        /// Encode with a specified block size using ArenaConfig.
        pub fn encodeWithBlockSizeArena(arena: ArenaConfig, values: []const ValueType, block_size: u16) ![]u8 {
            return encodeImpl(arena.temp, arena.output, values, block_size);
        }

        /// Encode with a specified block size.
        pub fn encodeWithBlockSize(allocator: std.mem.Allocator, values: []const ValueType, block_size: u16) ![]u8 {
            return encodeImpl(allocator, allocator, values, block_size);
        }

        fn encodeImpl(
            temp_alloc: std.mem.Allocator,
            out_alloc: std.mem.Allocator,
            values: []const ValueType,
            block_size: u16,
        ) ![]u8 {
            if (values.len == 0) {
                const data = try out_alloc.alloc(u8, HEADER_SIZE);
                std.mem.writeInt(u32, data[0..4], 0, .little);
                std.mem.writeInt(u16, data[4..6], block_size, .little);
                return data;
            }

            const n: u32 = @intCast(values.len);
            const num_blocks: u32 = (n + block_size - 1) / block_size;

            // Phase 1: Encode all deltas into a temporary buffer
            var delta_buf = std.ArrayListUnmanaged(u8){};
            defer delta_buf.deinit(temp_alloc);

            // Track byte offset of each block's deltas
            var block_offsets = try temp_alloc.alloc(u32, num_blocks);
            defer temp_alloc.free(block_offsets);

            for (0..num_blocks) |bi| {
                block_offsets[bi] = @intCast(delta_buf.items.len);

                const block_start: u32 = @intCast(bi * block_size);
                const block_end: u32 = @min(block_start + block_size, n);

                // First value in block is stored as absolute; encode deltas for the rest
                for (block_start + 1..block_end) |i| {
                    const delta = values[i] - values[i - 1];
                    try writeLeb128(&delta_buf, temp_alloc, delta);
                }
            }

            // Phase 2: Assemble final buffer
            // Layout: header + absolute_values + block_byte_offsets + delta_bytes
            const abs_section_size: u32 = num_blocks * value_size;
            const offsets_section_size: u32 = num_blocks * 4;
            const total_size: u32 = HEADER_SIZE + abs_section_size + offsets_section_size + @as(u32, @intCast(delta_buf.items.len));

            const data = try out_alloc.alloc(u8, total_size);

            // Header
            std.mem.writeInt(u32, data[0..4], n, .little);
            std.mem.writeInt(u16, data[4..6], block_size, .little);

            // Absolute values at block starts
            var abs_offset: u32 = HEADER_SIZE;
            for (0..num_blocks) |bi| {
                const idx: u32 = @intCast(bi * block_size);
                std.mem.writeInt(ValueType, data[abs_offset..][0..value_size], values[idx], .little);
                abs_offset += value_size;
            }

            // Block byte offsets
            var off_offset: u32 = HEADER_SIZE + abs_section_size;
            for (0..num_blocks) |bi| {
                std.mem.writeInt(u32, data[off_offset..][0..4], block_offsets[bi], .little);
                off_offset += 4;
            }

            // Delta bytes
            const delta_start = HEADER_SIZE + abs_section_size + offsets_section_size;
            @memcpy(data[delta_start..][0..delta_buf.items.len], delta_buf.items);

            return data;
        }

        /// Retrieve the value at `index` from encoded data.
        pub fn get(data: []const u8, index: u32) ValueType {
            const n = std.mem.readInt(u32, data[0..4], .little);
            _ = n;
            const block_size = std.mem.readInt(u16, data[4..6], .little);

            const block_idx = index / block_size;
            const idx_in_block = index % block_size;

            // Read absolute value at block start
            const abs_offset: u32 = HEADER_SIZE + block_idx * value_size;
            var value = std.mem.readInt(ValueType, data[abs_offset..][0..value_size], .little);

            if (idx_in_block == 0) return value;

            // Read block byte offset
            const num_blocks = (std.mem.readInt(u32, data[0..4], .little) + block_size - 1) / block_size;
            const offsets_start: u32 = HEADER_SIZE + num_blocks * value_size;
            const block_delta_offset = std.mem.readInt(u32, data[offsets_start + block_idx * 4 ..][0..4], .little);

            // Delta section start
            const delta_section_start: u32 = offsets_start + num_blocks * 4;
            var cursor: u32 = delta_section_start + block_delta_offset;

            // Decode deltas until we reach idx_in_block
            for (0..idx_in_block) |_| {
                const delta = readLeb128(data, &cursor);
                value += @intCast(delta);
            }

            return value;
        }

        /// Total number of encoded values.
        pub fn count(data: []const u8) u32 {
            return std.mem.readInt(u32, data[0..4], .little);
        }

        // ── LEB128 helpers ──

        fn writeLeb128(buf: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: ValueType) !void {
            var v: u64 = @intCast(value);
            while (true) {
                const byte: u8 = @intCast(v & 0x7F);
                v >>= 7;
                if (v == 0) {
                    try buf.append(allocator, byte);
                    break;
                } else {
                    try buf.append(allocator, byte | 0x80);
                }
            }
        }

        fn readLeb128(data: []const u8, cursor: *u32) u64 {
            var result: u64 = 0;
            var shift: u6 = 0;
            while (true) {
                const byte = data[cursor.*];
                cursor.* += 1;
                result |= @as(u64, byte & 0x7F) << shift;
                if (byte & 0x80 == 0) break;
                shift += 7;
            }
            return result;
        }
    };
}

// ── Tests ────────────────────────────────────────────────────
const testing = std.testing;
const DV = DeltaVarint(u32);

test "DeltaVarint: round-trip basic" {
    const values = [_]u32{ 100, 200, 300, 400, 500 };
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 5), DV.count(data));
    for (0..values.len) |i| {
        try testing.expectEqual(values[i], DV.get(data, @intCast(i)));
    }
}

test "DeltaVarint: monotone increasing offsets" {
    const values = [_]u32{ 0, 100, 250, 400, 600, 850, 1100, 1400 };
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (0..values.len) |i| {
        try testing.expectEqual(values[i], DV.get(data, @intCast(i)));
    }
}

test "DeltaVarint: constant deltas" {
    // 0, 10, 20, 30, ...
    var values: [100]u32 = undefined;
    for (0..100) |i| {
        values[i] = @intCast(i * 10);
    }
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (0..100) |i| {
        try testing.expectEqual(values[i], DV.get(data, @intCast(i)));
    }
}

test "DeltaVarint: single element" {
    const values = [_]u32{42};
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 1), DV.count(data));
    try testing.expectEqual(@as(u32, 42), DV.get(data, 0));
}

test "DeltaVarint: large jumps mixed with small" {
    const values = [_]u32{ 0, 1, 2, 3, 10000, 10001, 10002, 50000 };
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (0..values.len) |i| {
        try testing.expectEqual(values[i], DV.get(data, @intCast(i)));
    }
}

test "DeltaVarint: block boundary access" {
    // 128 values with block_size=64 → 2 blocks
    var values: [128]u32 = undefined;
    for (0..128) |i| {
        values[i] = @intCast(i * 5);
    }
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 64);
    defer testing.allocator.free(data);

    // Access first element of each block
    try testing.expectEqual(@as(u32, 0), DV.get(data, 0)); // block 0 start
    try testing.expectEqual(@as(u32, 320), DV.get(data, 64)); // block 1 start

    // Access last element of each block
    try testing.expectEqual(@as(u32, 315), DV.get(data, 63)); // block 0 end
    try testing.expectEqual(@as(u32, 635), DV.get(data, 127)); // block 1 end
}

test "DeltaVarint: random access doesn't require full decode" {
    var values: [200]u32 = undefined;
    for (0..200) |i| {
        values[i] = @intCast(i * 7 + 100);
    }
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    // Access last element directly
    try testing.expectEqual(values[199], DV.get(data, 199));
    // Access middle element
    try testing.expectEqual(values[100], DV.get(data, 100));
}

test "DeltaVarint: space savings" {
    // 100 values with small deltas should compress well
    var values: [100]u32 = undefined;
    for (0..100) |i| {
        values[i] = @intCast(i * 3);
    }
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    const naive_size = 100 * @sizeOf(u32);
    try testing.expect(data.len < naive_size);
}

test "DeltaVarint: empty input" {
    const values = [_]u32{};
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 0), DV.count(data));
}

test "DeltaVarint: u16 value type" {
    const DV16 = DeltaVarint(u16);
    const values = [_]u16{ 10, 20, 30, 40, 50 };
    const data = try DV16.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    for (0..values.len) |i| {
        try testing.expectEqual(values[i], DV16.get(data, @intCast(i)));
    }
}
