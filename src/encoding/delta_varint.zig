//! Delta encoding + variable-length integer encoding for sorted integer sequences.
//!
//! `DeltaVarint(ValueType)` is generic over the value width:
//!  - `u16` — small values (e.g. dictionary offsets within 64 KiB)
//!  - `u32` — standard 32-bit positions or offsets (most common)
//!  - `u64` — large values (e.g. file offsets, timestamps)
//!
//! Build-time: encode a monotone (non-decreasing) sequence into a compact byte
//! representation. The encoder rejects non-monotone input with `error.InvalidData`.
//! Read-time: random-access `get(index)` without decoding the entire sequence.
//!
//! Uses block-based layout with absolute values at block boundaries for random access.
//! LEB128 varint encoding for delta values (max 10 bytes = u64 ceiling).
//!
//! Internal types scale with `ValueType`:
//!  - `CountType` (element count, index): `u32` for ValueType ≤ u32, else `u64`
//!  - `OffsetType` (block byte offsets, blob size): `u32` for ValueType ≤ u32, else `u64`
//!
//! For untrusted data: call `validate()` before `get()` or `count()`. After a
//! successful `validate()`, all `get(i)` calls for `i < count()` are guaranteed
//! safe. Both `get()` and `count()` are unchecked — they assume valid data.
//!
//! Serialized format:
//!   [sizeof(CountType) bytes: count]              — number of encoded values (LE)
//!   [2 bytes: block_size (u16)]                    — elements per block
//!   [num_blocks × sizeof(ValueType) bytes: absolute values at block starts]
//!   [num_blocks × sizeof(OffsetType) bytes: byte offset into delta section for each block]
//!   [variable: LEB128-encoded deltas for each block]

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn DeltaVarint(comptime ValueType: type) type {
    comptime {
        const info = @typeInfo(ValueType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("DeltaVarint: ValueType must be an unsigned integer type, got " ++ @typeName(ValueType));
        if (@bitSizeOf(ValueType) < 16)
            @compileError("DeltaVarint: ValueType must be at least u16");
        if (@bitSizeOf(ValueType) > 64)
            @compileError("DeltaVarint: ValueType must be at most u64 (LEB128 ceiling)");
    }
    const value_size = @sizeOf(ValueType);

    // Count / index type: u32 for ValueType ≤ u32, u64 for u64.
    const CountType = if (@bitSizeOf(ValueType) <= 32) u32 else u64;
    const count_size = @sizeOf(CountType);

    // Offset type: controls max serialized blob size.
    // u32 for ValueType ≤ u32 (4 GiB ceiling), u64 for u64.
    const OffsetType = if (@bitSizeOf(ValueType) <= 32) u32 else u64;
    const offset_size = @sizeOf(OffsetType);

    return struct {
        pub const Error = error{
            InvalidData,
            Overflow,
        };

        /// Header: count(CountType) + block_size(u16).
        pub const HEADER_SIZE: u32 = count_size + 2;

        /// Encode a sorted sequence using ArenaConfig with default block size (64).
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
            if (block_size == 0) return error.InvalidData;

            if (values.len == 0) {
                const data = try out_alloc.alloc(u8, HEADER_SIZE);
                writeCount(data[0..count_size], 0);
                std.mem.writeInt(u16, data[count_size..][0..2], block_size, .little);
                return data;
            }

            if (values.len > std.math.maxInt(CountType)) return error.InvalidData;
            const n: CountType = @intCast(values.len);
            const num_blocks: CountType = (n + block_size - 1) / block_size;

            // Phase 1: Encode all deltas into a temporary buffer.
            var delta_buf = std.ArrayListUnmanaged(u8){};
            defer delta_buf.deinit(temp_alloc);

            var block_offsets = try temp_alloc.alloc(OffsetType, num_blocks);
            defer temp_alloc.free(block_offsets);

            for (0..num_blocks) |bi| {
                block_offsets[bi] = @intCast(delta_buf.items.len);

                const block_start: CountType = @intCast(@as(u64, bi) * block_size);
                const block_end: CountType = @min(block_start + block_size, n);

                for (block_start + 1..block_end) |i| {
                    if (values[i] < values[i - 1]) return error.InvalidData;
                    const delta = values[i] - values[i - 1];
                    try writeLeb128(&delta_buf, temp_alloc, delta);
                }
            }

            // Phase 2: Assemble final buffer.
            const abs_section_size_u64: u64 = @as(u64, num_blocks) * value_size;
            const offsets_section_size_u64: u64 = @as(u64, num_blocks) * offset_size;
            const total_size_u64: u64 = HEADER_SIZE + abs_section_size_u64 + offsets_section_size_u64 + delta_buf.items.len;
            if (total_size_u64 > std.math.maxInt(OffsetType)) return error.Overflow;

            const abs_section_size: OffsetType = @intCast(abs_section_size_u64);
            const offsets_section_size: OffsetType = @intCast(offsets_section_size_u64);
            const total_size: OffsetType = @intCast(total_size_u64);

            const data = try out_alloc.alloc(u8, total_size);

            // Header
            writeCount(data[0..count_size], n);
            std.mem.writeInt(u16, data[count_size..][0..2], block_size, .little);

            // Absolute values at block starts
            var abs_offset: OffsetType = HEADER_SIZE;
            for (0..num_blocks) |bi| {
                const idx: CountType = @intCast(@as(u64, bi) * block_size);
                std.mem.writeInt(ValueType, data[abs_offset..][0..value_size], values[idx], .little);
                abs_offset += value_size;
            }

            // Block byte offsets
            var off_offset: OffsetType = HEADER_SIZE + abs_section_size;
            for (0..num_blocks) |bi| {
                std.mem.writeInt(OffsetType, data[off_offset..][0..offset_size], block_offsets[bi], .little);
                off_offset += offset_size;
            }

            // Delta bytes
            const delta_start = HEADER_SIZE + abs_section_size + offsets_section_size;
            @memcpy(data[delta_start..][0..delta_buf.items.len], delta_buf.items);

            return data;
        }

        // ── Queries ──

        /// Retrieve the value at `index` from encoded data.
        /// Unchecked: assumes valid data. Use `validate()` first for untrusted data.
        pub fn get(data: []const u8, index: CountType) ValueType {
            const block_size: CountType = std.mem.readInt(u16, data[count_size..][0..2], .little);

            const block_idx = index / block_size;
            const idx_in_block = index % block_size;

            // Read absolute value at block start.
            const abs_offset: OffsetType = HEADER_SIZE + @as(OffsetType, @intCast(block_idx)) * value_size;
            var value = std.mem.readInt(ValueType, data[abs_offset..][0..value_size], .little);

            if (idx_in_block == 0) return value;

            // Read block byte offset.
            const n = readCount(data[0..count_size]);
            const num_blocks: CountType = (n + block_size - 1) / block_size;
            const offsets_start: OffsetType = HEADER_SIZE + @as(OffsetType, @intCast(num_blocks)) * value_size;
            const block_delta_offset = std.mem.readInt(OffsetType, data[offsets_start + @as(OffsetType, @intCast(block_idx)) * offset_size ..][0..offset_size], .little);

            // Delta section start.
            const delta_section_start: OffsetType = offsets_start + @as(OffsetType, @intCast(num_blocks)) * offset_size;
            var cursor: OffsetType = delta_section_start + block_delta_offset;

            for (0..idx_in_block) |_| {
                const delta = readLeb128(data, &cursor);
                value += @intCast(delta);
            }

            return value;
        }

        /// Total number of encoded values.
        /// Unchecked: assumes valid data. Use `validate()` first for untrusted data.
        pub fn count(data: []const u8) CountType {
            return readCount(data[0..count_size]);
        }

        /// Validate encoded data from an untrusted source.
        /// After a successful `validate()`, all `get(i)` for `i < count()` are
        /// guaranteed safe. Checks structural integrity (header, block offsets,
        /// offset-envelope), walks every LEB128 delta stream to verify
        /// well-formedness and termination, and verifies that all reconstructed
        /// values fit in `ValueType` and are monotonically non-decreasing.
        pub fn validate(data: []const u8) Error!void {
            if (data.len < HEADER_SIZE) return Error.InvalidData;

            const n = readCount(data[0..count_size]);
            const block_size: CountType = std.mem.readInt(u16, data[count_size..][0..2], .little);

            // block_size must be valid even for empty sequences (format invariant).
            if (block_size == 0) return Error.InvalidData;

            // Empty sequence: header-only is valid.
            if (n == 0) {
                if (data.len != HEADER_SIZE) return Error.InvalidData;
                return;
            }

            const num_blocks: u64 = (@as(u64, n) + block_size - 1) / block_size;
            const abs_section_size: u64 = num_blocks * value_size;
            const offsets_section_size: u64 = num_blocks * offset_size;
            const min_size: u64 = HEADER_SIZE + abs_section_size + offsets_section_size;

            if (data.len < min_size) return Error.InvalidData;

            // Enforce offset-envelope: total blob must fit in OffsetType.
            if (data.len > std.math.maxInt(OffsetType)) return Error.InvalidData;

            // Validate block offsets: within bounds and monotonically non-decreasing.
            const delta_section_len: u64 = data.len - min_size;
            const offsets_start: u64 = HEADER_SIZE + abs_section_size;
            var prev_block_offset: OffsetType = 0;

            for (0..num_blocks) |bi| {
                const off_pos = offsets_start + bi * offset_size;
                const block_offset = std.mem.readInt(OffsetType, data[@intCast(off_pos)..][0..offset_size], .little);
                if (block_offset > delta_section_len) return Error.InvalidData;
                if (bi > 0 and block_offset < prev_block_offset) return Error.InvalidData;
                prev_block_offset = block_offset;
            }

            // Walk each block's LEB128 stream to verify well-formedness and monotonicity.
            const delta_section_start: u64 = min_size;
            var prev_block_last_value: u64 = 0;

            for (0..num_blocks) |bi| {
                const block_start_idx: u64 = @as(u64, bi) * block_size;
                const block_end_idx: u64 = @min(block_start_idx + block_size, @as(u64, n));
                const expected_deltas = block_end_idx - block_start_idx - 1;

                // Delta byte range for this block.
                const off_pos = offsets_start + bi * offset_size;
                const blk_delta_start = delta_section_start + std.mem.readInt(OffsetType, data[@intCast(off_pos)..][0..offset_size], .little);
                const blk_delta_end: u64 = if (bi + 1 < num_blocks) blk: {
                    const next_off = offsets_start + (bi + 1) * offset_size;
                    break :blk delta_section_start + std.mem.readInt(OffsetType, data[@intCast(next_off)..][0..offset_size], .little);
                } else data.len;

                // Reconstruct values to verify deltas, overflow, and monotonicity.
                const abs_offset = HEADER_SIZE + @as(u64, bi) * value_size;
                var value: u64 = std.mem.readInt(ValueType, data[@intCast(abs_offset)..][0..value_size], .little);

                // Cross-block monotonicity: block N's anchor >= block N-1's last value.
                if (bi > 0 and value < prev_block_last_value) return Error.InvalidData;

                var pos: usize = @intCast(blk_delta_start);
                const end: usize = @intCast(blk_delta_end);

                for (0..expected_deltas) |_| {
                    const delta = validateLeb128(data, &pos, end) orelse return Error.InvalidData;
                    const sum = @addWithOverflow(value, delta);
                    if (sum[1] != 0) return Error.InvalidData;
                    value = sum[0];
                    if (value > std.math.maxInt(ValueType)) return Error.InvalidData;
                }

                if (pos != end) return Error.InvalidData;

                prev_block_last_value = value;
            }
        }

        // ── Internal helpers ──

        fn writeCount(buf: *[count_size]u8, val: CountType) void {
            std.mem.writeInt(CountType, buf, val, .little);
        }

        fn readCount(buf: *const [count_size]u8) CountType {
            return std.mem.readInt(CountType, buf, .little);
        }

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

        /// Decode a single LEB128 value. Bounded to 10 bytes (max for u64)
        /// and guards against reading past the end of `data`.
        fn readLeb128(data: []const u8, cursor: *OffsetType) u64 {
            var result: u64 = 0;
            inline for (0..10) |i| {
                if (cursor.* >= data.len) return result;
                const byte = data[cursor.*];
                cursor.* += 1;
                if (i == 9) {
                    // 10th byte: only bit 0 is valid for u64 (bits 63).
                    result |= @as(u64, byte & 0x01) << 63;
                } else {
                    result |= @as(u64, byte & 0x7F) << @intCast(i * 7);
                }
                if (byte & 0x80 == 0) return result;
            }
            return result;
        }

        /// Validate a single LEB128 value, returning the decoded value or null
        /// if the encoding is malformed (truncated, overlong, or has invalid
        /// bits in the 10th byte).
        fn validateLeb128(data: []const u8, pos: *usize, end: usize) ?u64 {
            var result: u64 = 0;
            inline for (0..10) |i| {
                if (pos.* >= end) return null;
                const byte = data[pos.*];
                pos.* += 1;
                if (i == 9) {
                    // 10th byte: only bit 0 is valid payload for u64.
                    // Reject if higher bits are set (non-canonical encoding).
                    if (byte & 0x7E != 0) return null;
                    result |= @as(u64, byte & 0x01) << 63;
                } else {
                    result |= @as(u64, byte & 0x7F) << @intCast(i * 7);
                }
                if (byte & 0x80 == 0) return result;
            }
            return null; // Overlong: >10 continuation bytes
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

test "DeltaVarint: u64 value type" {
    const DV64 = DeltaVarint(u64);
    const values = [_]u64{ 100_000_000_000, 200_000_000_000, 300_000_000_000 };
    const data = try DV64.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u64, 3), DV64.count(data));
    for (0..values.len) |i| {
        try testing.expectEqual(values[i], DV64.get(data, @intCast(i)));
    }
}

test "DeltaVarint: validate accepts valid data" {
    const values = [_]u32{ 10, 20, 30, 40, 50 };
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    try DV.validate(data);
}

test "DeltaVarint: validate accepts empty" {
    const values = [_]u32{};
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    try DV.validate(data);
}

test "DeltaVarint: validate rejects too-short blob" {
    const short = [_]u8{ 0, 0, 0 };
    try testing.expectError(DV.Error.InvalidData, DV.validate(&short));
}

test "DeltaVarint: validate rejects truncated payload" {
    const values = [_]u32{ 10, 20, 30, 40, 50 };
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    // Truncate: keep header but chop the body
    try testing.expectError(DV.Error.InvalidData, DV.validate(data[0..DV.HEADER_SIZE]));
}

test "DeltaVarint: validate rejects corrupted LEB128 (all continuation bytes)" {
    // Craft a blob where the delta section is all continuation bytes (0x80).
    // This should fail validation because the LEB128 never terminates.
    const values = [_]u32{ 0, 10 };
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    // Copy and corrupt the delta section with unterminated LEB128.
    var corrupt = try testing.allocator.alloc(u8, data.len);
    defer testing.allocator.free(corrupt);
    @memcpy(corrupt, data);

    // Delta section starts after header + abs_section + offsets_section.
    // 1 block: header(6) + 1*4(abs) + 1*4(offsets) = 14, delta starts at 14.
    const delta_start: usize = DV.HEADER_SIZE + 4 + 4;
    for (delta_start..corrupt.len) |i| {
        corrupt[i] = 0x80; // All continuation bits set, never terminates
    }
    try testing.expectError(DV.Error.InvalidData, DV.validate(corrupt));
}

test "DeltaVarint: validate rejects delta overflow for ValueType" {
    // Craft a blob where a LEB128 delta would overflow u32 when added.
    const values = [_]u32{ std.math.maxInt(u32) - 1, std.math.maxInt(u32) };
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    // Valid: delta is 1, abs is maxInt-1, sum is maxInt.
    try DV.validate(data);

    // Now corrupt: change absolute value to maxInt and keep delta=1 → overflow.
    var corrupt = try testing.allocator.alloc(u8, data.len);
    defer testing.allocator.free(corrupt);
    @memcpy(corrupt, data);

    // Absolute value at block start is at HEADER_SIZE.
    std.mem.writeInt(u32, corrupt[DV.HEADER_SIZE..][0..4], std.math.maxInt(u32), .little);
    try testing.expectError(DV.Error.InvalidData, DV.validate(corrupt));
}

test "DeltaVarint: block_size=1 round-trip" {
    // Every element gets its own block: delta section is empty.
    const values = [_]u32{ 100, 300, 500, 700 };
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 1);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 4), DV.count(data));
    for (0..values.len) |i| {
        try testing.expectEqual(values[i], DV.get(data, @intCast(i)));
    }
    try DV.validate(data);
}

test "DeltaVarint: many consecutive duplicates" {
    // 100 identical values: all deltas are zero.
    var values: [100]u32 = undefined;
    @memset(&values, 42);
    const data = try DV.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 100), DV.count(data));
    for (0..100) |i| {
        try testing.expectEqual(@as(u32, 42), DV.get(data, @intCast(i)));
    }
    try DV.validate(data);
}

test "DeltaVarint: block_size=0 rejected" {
    const values = [_]u32{ 1, 2, 3 };
    try testing.expectError(error.InvalidData, DV.encodeWithBlockSize(testing.allocator, &values, 0));
}

test "DeltaVarint: validate rejects block_offset monotonicity violation" {
    // Encode valid data, then corrupt block offsets to be non-monotonic.
    var values: [200]u32 = undefined;
    for (0..200) |i| values[i] = @intCast(i * 10);
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 64);
    defer testing.allocator.free(data);

    // Should be 4 blocks (200/64 = 3.125 → 4 blocks).
    var corrupt = try testing.allocator.alloc(u8, data.len);
    defer testing.allocator.free(corrupt);
    @memcpy(corrupt, data);

    // Offsets section: header(6) + 4 blocks * 4 bytes abs = 22, offsets at 22.
    const offsets_start: usize = DV.HEADER_SIZE + 4 * @sizeOf(u32);
    // Set block 1's offset to 0 (less than block 0's end).
    // Block 0 offset should be 0, block 1 should be > 0. Set block 1 to 0.
    // Actually block 0 is always 0, so set block 2 offset < block 1 offset.
    const block1_offset = std.mem.readInt(u32, corrupt[offsets_start + 4 ..][0..4], .little);
    if (block1_offset > 0) {
        // Set block 2 offset to less than block 1.
        std.mem.writeInt(u32, corrupt[offsets_start + 8 ..][0..4], block1_offset - 1, .little);
        try testing.expectError(DV.Error.InvalidData, DV.validate(corrupt));
    }
}

test "DeltaVarint: encode rejects non-monotone input" {
    const descending = [_]u32{ 10, 5, 1 };
    try testing.expectError(error.InvalidData, DV.encode(testing.allocator, &descending));
}

test "DeltaVarint: encode rejects non-monotone across block boundary" {
    // 5 values with block_size=2 → 3 blocks. Non-monotone at index 3.
    const values = [_]u32{ 10, 20, 30, 25, 40 };
    try testing.expectError(error.InvalidData, DV.encodeWithBlockSize(testing.allocator, &values, 2));
}

test "DeltaVarint: validate rejects block_size=0 on empty blob" {
    // Craft a header-only blob with block_size=0.
    var blob: [DV.HEADER_SIZE]u8 = undefined;
    std.mem.writeInt(u32, blob[0..4], 0, .little); // count = 0
    std.mem.writeInt(u16, blob[4..6], 0, .little); // block_size = 0
    try testing.expectError(DV.Error.InvalidData, DV.validate(&blob));
}

test "DeltaVarint: validate rejects cross-block monotonicity violation" {
    // Encode valid data, then corrupt the second block's absolute value
    // to be less than the last reconstructed value of the first block.
    var values: [128]u32 = undefined;
    for (0..128) |i| values[i] = @intCast(i * 10);
    const data = try DV.encodeWithBlockSize(testing.allocator, &values, 64);
    defer testing.allocator.free(data);

    var corrupt = try testing.allocator.alloc(u8, data.len);
    defer testing.allocator.free(corrupt);
    @memcpy(corrupt, data);

    // Block 1's absolute value is at HEADER_SIZE + 1 * sizeof(u32).
    // First block's last value is values[63] = 630. Set block 1's anchor to 0.
    const block1_abs: usize = DV.HEADER_SIZE + @sizeOf(u32);
    std.mem.writeInt(u32, corrupt[block1_abs..][0..4], 0, .little);
    try testing.expectError(DV.Error.InvalidData, DV.validate(corrupt));
}

test "DeltaVarint: u64 OffsetType round-trip" {
    const DV64 = DeltaVarint(u64);
    const values = [_]u64{ 1, 100, 200, 300, 1_000_000_000_000 };
    const data = try DV64.encode(testing.allocator, &values);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u64, 5), DV64.count(data));
    for (0..values.len) |i| {
        try testing.expectEqual(values[i], DV64.get(data, @intCast(i)));
    }
    try DV64.validate(data);
}
