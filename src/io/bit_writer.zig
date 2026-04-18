//! Bit-level output buffer that accumulates bits and produces a `[]u8`.
//!
//! Uses an allocator for buffer growth. MSB-first bit ordering.
//! Paired with `BitReader` for round-trip encode/decode workflows.
//!
//! ```
//! var writer = BitWriter(u32).init(allocator);
//! defer writer.deinit();
//! try writer.writeBits(0b1011, 4);
//! const bytes = try writer.finish();
//! ```

const std = @import("std");

pub fn BitWriter(comptime OffsetType: type) type {
    comptime {
        const info = @typeInfo(OffsetType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("BitWriter: OffsetType must be an unsigned integer type, got " ++ @typeName(OffsetType));
        if (@sizeOf(OffsetType) < 2)
            @compileError("BitWriter: OffsetType must be at least u16");
    }
    return struct {
        const Self = @This();

        buffer: std.ArrayListUnmanaged(u8),
        allocator: std.mem.Allocator,
        current_byte: u8,
        bits_in_current: u4, // 0–8 (u4 fits 0..15, we use 0..8)
        total_bits: OffsetType,

        pub const Error = std.mem.Allocator.Error;

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .buffer = .{},
                .allocator = allocator,
                .current_byte = 0,
                .bits_in_current = 0,
                .total_bits = 0,
            };
        }

        /// Write `n` bits from `value`. MSB-first ordering.
        /// The lowest `n` bits of `value` are written, with the highest of those first.
        pub fn writeBits(self: *Self, value: u64, n: u6) Error!void {
            if (n == 0) return;

            var remaining: u6 = n;
            while (remaining > 0) {
                const space_in_byte: u4 = 8 - self.bits_in_current;
                const bits_to_write: u4 = if (remaining < space_in_byte) @intCast(remaining) else space_in_byte;

                // Extract the next `bits_to_write` bits from the MSB end of what remains.
                // remaining-1 is the index of the highest bit we haven't written yet.
                const shift: u6 = remaining - @as(u6, bits_to_write);
                const mask: u64 = (@as(u64, 1) << bits_to_write) - 1;
                const extracted: u8 = @intCast((value >> shift) & mask);

                // Place extracted bits into current_byte at the correct position.
                const dest_shift: u3 = @intCast(space_in_byte - bits_to_write);
                self.current_byte |= extracted << dest_shift;
                self.bits_in_current += bits_to_write;
                self.total_bits += bits_to_write;
                remaining -= @as(u6, bits_to_write);

                // Flush full byte to buffer.
                if (self.bits_in_current == 8) {
                    try self.buffer.append(self.allocator, self.current_byte);
                    self.current_byte = 0;
                    self.bits_in_current = 0;
                }
            }
        }

        /// Write a single bit. MSB-first.
        pub fn writeBit(self: *Self, bit: u1) Error!void {
            const space_in_byte: u4 = 8 - self.bits_in_current;
            self.current_byte |= @as(u8, bit) << @intCast(space_in_byte - 1);
            self.bits_in_current += 1;
            self.total_bits += 1;

            if (self.bits_in_current == 8) {
                try self.buffer.append(self.allocator, self.current_byte);
                self.current_byte = 0;
                self.bits_in_current = 0;
            }
        }

        /// Flush partial byte (zero-padded) and return the completed buffer.
        /// Caller owns the returned slice.
        pub fn finish(self: *Self) Error![]u8 {
            if (self.bits_in_current > 0) {
                // Partial last byte — already zero-padded because current_byte
                // starts as 0 and we only OR bits in.
                try self.buffer.append(self.allocator, self.current_byte);
                self.current_byte = 0;
                self.bits_in_current = 0;
            }
            return self.buffer.toOwnedSlice(self.allocator);
        }

        /// Current write position in bits.
        pub fn bitPosition(self: Self) OffsetType {
            return self.total_bits;
        }

        /// Pad with zero bits to the next byte boundary.
        pub fn alignToByte(self: *Self) Error!void {
            if (self.bits_in_current > 0) {
                // Flush partial byte (remaining bits are already zero).
                try self.buffer.append(self.allocator, self.current_byte);
                self.total_bits += 8 - @as(OffsetType, self.bits_in_current);
                self.current_byte = 0;
                self.bits_in_current = 0;
            }
        }

        pub fn deinit(self: *Self) void {
            self.buffer.deinit(self.allocator);
        }
    };
}

// ── Tests ────────────────────────────────────────────────────
const testing = std.testing;
const BitReader = @import("bit_reader.zig").BitReader(u32); // same directory

test "BitWriter: init and position" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();
    try testing.expectEqual(@as(u32, 0), writer.bitPosition());
}

test "BitWriter: writeBit single bits" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    // Write 1,0,1,1,0,0,0,0 = 0xB0
    try writer.writeBit(1);
    try writer.writeBit(0);
    try writer.writeBit(1);
    try writer.writeBit(1);
    try writer.writeBit(0);
    try writer.writeBit(0);
    try writer.writeBit(0);
    try writer.writeBit(0);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(usize, 1), bytes.len);
    try testing.expectEqual(@as(u8, 0xB0), bytes[0]);
}

test "BitWriter: writeBits 4 bits" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    try writer.writeBits(0b1010, 4);
    try writer.writeBits(0b1011, 4);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(usize, 1), bytes.len);
    try testing.expectEqual(@as(u8, 0xAB), bytes[0]);
}

test "BitWriter: cross-byte write (12 bits)" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    // Write 0xABC = 0b1010_1011_1100 as 12 bits
    try writer.writeBits(0xABC, 12);
    try writer.writeBits(0xD, 4);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(usize, 2), bytes.len);
    try testing.expectEqual(@as(u8, 0xAB), bytes[0]);
    try testing.expectEqual(@as(u8, 0xCD), bytes[1]);
}

test "BitWriter: finish pads partial byte with zeros" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    // Write 3 bits: 101 → should produce 0b10100000 = 0xA0
    try writer.writeBits(0b101, 3);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(usize, 1), bytes.len);
    try testing.expectEqual(@as(u8, 0xA0), bytes[0]);
}

test "BitWriter: finish on empty writer" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(usize, 0), bytes.len);
}

test "BitWriter: writeBits with n=0" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    try writer.writeBits(0xFF, 0); // should be no-op
    try testing.expectEqual(@as(u32, 0), writer.bitPosition());
}

test "BitWriter: bitPosition tracks correctly" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    try writer.writeBits(0, 5);
    try testing.expectEqual(@as(u32, 5), writer.bitPosition());

    try writer.writeBits(0, 8);
    try testing.expectEqual(@as(u32, 13), writer.bitPosition());

    try writer.writeBit(0);
    try testing.expectEqual(@as(u32, 14), writer.bitPosition());
}

test "BitWriter: alignToByte" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    try writer.writeBits(0b111, 3);
    try testing.expectEqual(@as(u32, 3), writer.bitPosition());

    try writer.alignToByte();
    try testing.expectEqual(@as(u32, 8), writer.bitPosition());

    // Write a full byte after alignment
    try writer.writeBits(0xAA, 8);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(usize, 2), bytes.len);
    try testing.expectEqual(@as(u8, 0xE0), bytes[0]); // 111_00000
    try testing.expectEqual(@as(u8, 0xAA), bytes[1]);
}

test "BitWriter: alignToByte when already aligned" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    try writer.writeBits(0xFF, 8); // exactly 1 byte
    try writer.alignToByte(); // no-op
    try testing.expectEqual(@as(u32, 8), writer.bitPosition());
}

test "BitWriter: sequential writes concatenate correctly" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    try writer.writeBits(0xFF, 8);
    try writer.writeBits(0x00, 8);
    try writer.writeBits(0xAA, 8);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(usize, 3), bytes.len);
    try testing.expectEqual(@as(u8, 0xFF), bytes[0]);
    try testing.expectEqual(@as(u8, 0x00), bytes[1]);
    try testing.expectEqual(@as(u8, 0xAA), bytes[2]);
}

test "BitWriter → BitReader round-trip" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    // Write various bit widths
    try writer.writeBits(0b1011, 4); // 4 bits
    try writer.writeBits(0xDEAD, 16); // 16 bits
    try writer.writeBits(0b110, 3); // 3 bits
    try writer.writeBit(1); // 1 bit

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    var reader = BitReader.init(bytes);
    try testing.expectEqual(@as(u4, 0b1011), try reader.readBits(u4, 4));
    try testing.expectEqual(@as(u16, 0xDEAD), try reader.readBits(u16, 16));
    try testing.expectEqual(@as(u3, 0b110), try reader.readBits(u3, 3));
    try testing.expectEqual(@as(u1, 1), try reader.readBit());
}

test "BitWriter → BitReader round-trip: many small writes" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    // Write 100 single bits: alternating 1,0,1,0,...
    for (0..100) |i| {
        try writer.writeBit(@intCast(i % 2));
    }

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    var reader = BitReader.init(bytes);
    for (0..100) |i| {
        const expected: u1 = @intCast(i % 2);
        const got = try reader.readBit();
        try testing.expectEqual(expected, got);
    }
}

test "BitWriter: write full 32-bit value" {
    var writer = BitWriter(u32).init(testing.allocator);
    defer writer.deinit();

    try writer.writeBits(0xDEADBEEF, 32);

    const bytes = try writer.finish();
    defer testing.allocator.free(bytes);

    try testing.expectEqual(@as(usize, 4), bytes.len);
    try testing.expectEqual(@as(u8, 0xDE), bytes[0]);
    try testing.expectEqual(@as(u8, 0xAD), bytes[1]);
    try testing.expectEqual(@as(u8, 0xBE), bytes[2]);
    try testing.expectEqual(@as(u8, 0xEF), bytes[3]);
}
