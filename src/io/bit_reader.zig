//! Bit-level read cursor over a byte slice.
//!
//! Provides sequential and random-access bit reads from a borrowed `[]const u8`.
//! MSB-first bit ordering. Zero allocation — operates entirely on the provided slice.
//!
//! ```
//! var reader = BitReader.init(data);
//! const value = try reader.readBits(u12, 12);
//! ```

pub const BitReader = struct {
    data: []const u8,
    bit_pos: u32,

    pub const Error = error{EndOfStream};

    pub fn init(data: []const u8) BitReader {
        return .{ .data = data, .bit_pos = 0 };
    }

    /// Read `n` bits and return them as type `T`. MSB-first ordering.
    /// Returns `error.EndOfStream` if fewer than `n` bits remain.
    pub fn readBits(self: *BitReader, comptime T: type, n: u6) Error!T {
        const bits: u32 = n;
        if (bits == 0) return 0;
        if (self.bit_pos + bits > self.data.len * 8) return error.EndOfStream;

        var result: u64 = 0;
        var remaining_bits: u32 = bits;
        var cur_pos = self.bit_pos;

        while (remaining_bits > 0) {
            const byte_index = cur_pos / 8;
            const bit_offset: u3 = @intCast(cur_pos % 8);
            const bits_available_in_byte: u32 = 8 - @as(u32, bit_offset);
            const bits_to_read = @min(bits_available_in_byte, remaining_bits);

            // Extract bits from current byte, MSB-first.
            // Shift the byte right so the target bits are in the LSB positions,
            // then mask off just the bits we want.
            const shift: u3 = @intCast(bits_available_in_byte - bits_to_read);
            const mask: u8 = @intCast((@as(u16, 1) << @intCast(bits_to_read)) - 1);
            const extracted: u64 = (self.data[byte_index] >> shift) & mask;

            result = (result << @intCast(bits_to_read)) | extracted;
            cur_pos += bits_to_read;
            remaining_bits -= bits_to_read;
        }

        self.bit_pos = cur_pos;

        const type_info = @typeInfo(T);
        const type_bits: u6 = switch (type_info) {
            .int => |i| i.bits,
            else => @compileError("readBits requires an integer type"),
        };
        if (type_bits >= 64) {
            return @as(T, @intCast(result));
        }
        return @as(T, @truncate(result));
    }

    /// Read a single bit. MSB-first.
    pub fn readBit(self: *BitReader) Error!u1 {
        if (self.bit_pos >= self.data.len * 8) return error.EndOfStream;

        const byte_index = self.bit_pos / 8;
        const bit_offset: u3 = @intCast(self.bit_pos % 8);

        // MSB-first: bit 0 of a byte is the most significant bit.
        const bit: u1 = @intCast((self.data[byte_index] >> (7 - @as(u3, bit_offset))) & 1);
        self.bit_pos += 1;
        return bit;
    }

    /// Read `n` bits without bounds checking. Caller guarantees sufficient data.
    pub fn readBitsUnchecked(self: *BitReader, comptime T: type, n: u6) T {
        return self.readBits(T, n) catch unreachable;
    }

    /// Read a single bit without bounds checking.
    pub fn readBitUnchecked(self: *BitReader) u1 {
        return self.readBit() catch unreachable;
    }

    /// Seek to an absolute bit position.
    pub fn seekToBit(self: *BitReader, target: u32) void {
        self.bit_pos = target;
    }

    /// Number of bits remaining from cursor to end of data.
    pub fn remaining(self: BitReader) u32 {
        const total: u32 = @intCast(self.data.len * 8);
        return if (self.bit_pos >= total) 0 else total - self.bit_pos;
    }

    /// Advance cursor to the next byte boundary.
    pub fn alignToByte(self: *BitReader) void {
        const rem = self.bit_pos % 8;
        if (rem != 0) self.bit_pos += 8 - rem;
    }

    /// Number of bytes consumed so far (rounded up).
    pub fn bytesRead(self: BitReader) u32 {
        return (self.bit_pos + 7) / 8;
    }

    /// Current bit position.
    pub fn bitPos(self: BitReader) u32 {
        return self.bit_pos;
    }

    /// Whether the cursor has reached the end of the data.
    pub fn isAtEnd(self: BitReader) bool {
        return self.bit_pos >= self.data.len * 8;
    }
};

// ── Tests ────────────────────────────────────────────────────
const testing = @import("std").testing;

test "BitReader: init and remaining" {
    const data = [_]u8{ 0xAB, 0xCD };
    const reader = BitReader.init(&data);
    try testing.expectEqual(@as(u32, 16), reader.remaining());
}

test "BitReader: readBit MSB-first order" {
    // 0b10110000 = 0xB0
    const data = [_]u8{0xB0};
    var reader = BitReader.init(&data);

    // First 4 bits of 0xB0 = 1, 0, 1, 1
    try testing.expectEqual(@as(u1, 1), try reader.readBit());
    try testing.expectEqual(@as(u1, 0), try reader.readBit());
    try testing.expectEqual(@as(u1, 1), try reader.readBit());
    try testing.expectEqual(@as(u1, 1), try reader.readBit());

    // Next 4 bits = 0, 0, 0, 0
    try testing.expectEqual(@as(u1, 0), try reader.readBit());
    try testing.expectEqual(@as(u1, 0), try reader.readBit());
    try testing.expectEqual(@as(u1, 0), try reader.readBit());
    try testing.expectEqual(@as(u1, 0), try reader.readBit());
}

test "BitReader: readBits 1-8 bits" {
    // 0xAB = 0b10101011
    const data = [_]u8{0xAB};
    var reader = BitReader.init(&data);

    // Read 4 bits: 0b1010 = 10
    try testing.expectEqual(@as(u4, 0b1010), try reader.readBits(u4, 4));
    // Read 4 bits: 0b1011 = 11
    try testing.expectEqual(@as(u4, 0b1011), try reader.readBits(u4, 4));
}

test "BitReader: cross-byte reads" {
    // 0xAB = 10101011, 0xCD = 11001101
    // Reading 12 bits from bit 0: 10101011_1100 = 0xABC
    const data = [_]u8{ 0xAB, 0xCD };
    var reader = BitReader.init(&data);

    const val = try reader.readBits(u12, 12);
    try testing.expectEqual(@as(u12, 0xABC), val);

    // 4 bits remaining: 1101 = 0xD
    try testing.expectEqual(@as(u4, 0xD), try reader.readBits(u4, 4));
}

test "BitReader: sequential reads advance cursor" {
    const data = [_]u8{ 0xFF, 0x00, 0xAA };
    var reader = BitReader.init(&data);

    _ = try reader.readBits(u8, 8); // consume byte 0
    try testing.expectEqual(@as(u32, 8), reader.bitPos());
    try testing.expectEqual(@as(u32, 16), reader.remaining());

    _ = try reader.readBits(u8, 8); // consume byte 1
    try testing.expectEqual(@as(u32, 16), reader.bitPos());
    try testing.expectEqual(@as(u32, 8), reader.remaining());

    _ = try reader.readBits(u8, 8); // consume byte 2
    try testing.expect(reader.isAtEnd());
    try testing.expectEqual(@as(u32, 0), reader.remaining());
}

test "BitReader: seek + read" {
    // 0xDE = 11011110, 0xAD = 10101101
    const data = [_]u8{ 0xDE, 0xAD };
    var reader = BitReader.init(&data);

    // Read from bit 4 (middle of first byte)
    reader.seekToBit(4);
    // Bits 4..11 = 1110_1010 = 0xEA
    const val = try reader.readBits(u8, 8);
    try testing.expectEqual(@as(u8, 0xEA), val);
}

test "BitReader: end of stream error" {
    const data = [_]u8{0xFF};
    var reader = BitReader.init(&data);

    _ = try reader.readBits(u8, 8); // consume all 8 bits
    const result = reader.readBit();
    try testing.expectError(error.EndOfStream, result);
}

test "BitReader: read past end returns error" {
    const data = [_]u8{0xFF};
    var reader = BitReader.init(&data);

    // Try to read 16 bits from 8-bit data
    const result = reader.readBits(u16, 16);
    try testing.expectError(error.EndOfStream, result);
}

test "BitReader: MSB-first convention (0b10110000)" {
    // Requirement from spec: Bit `0b10110000` in byte 0 → first 4 bits read as 0b1011 = 11
    const data = [_]u8{0b10110000};
    var reader = BitReader.init(&data);

    const first_four = try reader.readBits(u4, 4);
    try testing.expectEqual(@as(u4, 0b1011), first_four);
    try testing.expectEqual(@as(u4, 11), first_four);
}

test "BitReader: alignToByte" {
    const data = [_]u8{ 0xFF, 0xAA };
    var reader = BitReader.init(&data);

    _ = try reader.readBits(u3, 3); // read 3 bits
    try testing.expectEqual(@as(u32, 3), reader.bitPos());

    reader.alignToByte(); // skip to bit 8
    try testing.expectEqual(@as(u32, 8), reader.bitPos());

    // Reading from byte 1 should give 0xAA
    try testing.expectEqual(@as(u8, 0xAA), try reader.readBits(u8, 8));
}

test "BitReader: alignToByte when already aligned" {
    const data = [_]u8{0xFF};
    var reader = BitReader.init(&data);
    reader.alignToByte(); // already at bit 0, should stay
    try testing.expectEqual(@as(u32, 0), reader.bitPos());
}

test "BitReader: bytesRead" {
    const data = [_]u8{ 0xFF, 0xFF };
    var reader = BitReader.init(&data);

    try testing.expectEqual(@as(u32, 0), reader.bytesRead());
    _ = try reader.readBit();
    try testing.expectEqual(@as(u32, 1), reader.bytesRead()); // 1 bit → rounds up to 1 byte
    _ = try reader.readBits(u7, 7);
    try testing.expectEqual(@as(u32, 1), reader.bytesRead()); // 8 bits = exactly 1 byte
    _ = try reader.readBit();
    try testing.expectEqual(@as(u32, 2), reader.bytesRead()); // 9 bits → rounds up to 2 bytes
}

test "BitReader: readBits with n=0" {
    const data = [_]u8{0xFF};
    var reader = BitReader.init(&data);
    const val = try reader.readBits(u8, 0);
    try testing.expectEqual(@as(u8, 0), val);
    try testing.expectEqual(@as(u32, 0), reader.bitPos()); // cursor shouldn't advance
}

test "BitReader: empty data" {
    const data = [_]u8{};
    var reader = BitReader.init(&data);
    try testing.expectEqual(@as(u32, 0), reader.remaining());
    try testing.expect(reader.isAtEnd());
    try testing.expectError(error.EndOfStream, reader.readBit());
}

test "BitReader: read full 16-bit value" {
    const data = [_]u8{ 0xDE, 0xAD };
    var reader = BitReader.init(&data);
    try testing.expectEqual(@as(u16, 0xDEAD), try reader.readBits(u16, 16));
}

test "BitReader: read full 32-bit value" {
    const data = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF };
    var reader = BitReader.init(&data);
    try testing.expectEqual(@as(u32, 0xDEADBEEF), try reader.readBits(u32, 32));
}
