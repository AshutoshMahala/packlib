//! Move-to-Front (MTF) Transform — locality-exploiting symbol transform.
//!
//! Transforms a symbol sequence such that frequently recurring symbols get
//! mapped to small indices (near zero). Typically used after BWT to
//! create a highly skewed distribution suitable for entropy coding.
//!
//! **Forward:** O(n × σ) where σ = alphabet size = `1 << @bitSizeOf(SymbolType)`.
//! **Inverse:** O(n × σ), identical cost.
//!
//! ## Generic parameter
//!
//! `Mtf(SymbolType)` — `SymbolType` controls the alphabet width:
//!   - `u4`  — 16-symbol nibble MTF (pairs with nibble-wide rANS/tANS)
//!   - `u8`  — 256-symbol byte MTF (classic BWT pipeline)
//!   - `u16` — 65 536-symbol token MTF (e.g. word/BPE-token sequences)
//!
//! Output indices are returned as `SymbolType` (max index = `2^bits - 1`,
//! which fits exactly).
//!
//! ## Algorithm
//!
//! Maintain a list of all `1 << @bitSizeOf(SymbolType)` symbol values. For each
//! input symbol:
//!   1. Output its current index in the list.
//!   2. Move it to the front of the list.
//!
//! After BWT, runs of the same symbol become runs of zero, and symbols from
//! similar contexts produce small indices.
//!
//! ## References
//!
//! - Bentley et al. (1986), "A locally adaptive data compression scheme"
//! - Burrows & Wheeler (1994), combined with BWT for compression

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

/// Move-to-Front transform parameterized by `SymbolType`.
pub fn Mtf(comptime SymbolType: type) type {
    comptime {
        const info = @typeInfo(SymbolType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("Mtf: SymbolType must be an unsigned integer type, got " ++ @typeName(SymbolType));
        const bits = @bitSizeOf(SymbolType);
        if (bits == 0)
            @compileError("Mtf: SymbolType must have at least 1 bit");
        if (bits > 16)
            @compileError("Mtf: SymbolType must be at most 16 bits (alphabet up to 65 536)");
    }

    const ALPHABET_SIZE: u32 = 1 << @bitSizeOf(SymbolType);

    return struct {
        // ── Forward Transform ──

        /// Forward MTF transform: symbols → MTF indices.
        /// Result allocated from `arena.output`. Input is not modified.
        pub fn forward(arena: ArenaConfig, input: []const SymbolType) ![]SymbolType {
            return forwardImpl(arena.output, input);
        }

        /// Forward MTF transform with a single allocator.
        pub fn forwardUniform(allocator: std.mem.Allocator, input: []const SymbolType) ![]SymbolType {
            return forwardImpl(allocator, input);
        }

        fn forwardImpl(allocator: std.mem.Allocator, input: []const SymbolType) ![]SymbolType {
            const output = try allocator.alloc(SymbolType, input.len);
            forwardInPlace(input, output);
            return output;
        }

        /// Forward MTF into a caller-provided buffer. No allocation.
        /// `output` must be at least `input.len` symbols.
        pub fn forwardInPlace(input: []const SymbolType, output: []SymbolType) void {
            var list: [ALPHABET_SIZE]SymbolType = undefined;
            for (0..ALPHABET_SIZE) |i| list[i] = @intCast(i);

            for (input, 0..) |sym, i| {
                const idx = findInList(&list, sym);
                output[i] = idx;
                moveToFront(&list, idx);
            }
        }

        // ── Inverse Transform ──

        /// Inverse MTF transform: MTF indices → original symbols.
        pub fn inverse(arena: ArenaConfig, input: []const SymbolType) ![]SymbolType {
            return inverseImpl(arena.output, input);
        }

        /// Inverse MTF transform with a single allocator.
        pub fn inverseUniform(allocator: std.mem.Allocator, input: []const SymbolType) ![]SymbolType {
            return inverseImpl(allocator, input);
        }

        fn inverseImpl(allocator: std.mem.Allocator, input: []const SymbolType) ![]SymbolType {
            const output = try allocator.alloc(SymbolType, input.len);
            inverseInPlace(input, output);
            return output;
        }

        /// Inverse MTF into a caller-provided buffer. No allocation.
        pub fn inverseInPlace(input: []const SymbolType, output: []SymbolType) void {
            var list: [ALPHABET_SIZE]SymbolType = undefined;
            for (0..ALPHABET_SIZE) |i| list[i] = @intCast(i);

            for (input, 0..) |idx, i| {
                const sym = list[idx];
                output[i] = sym;
                moveToFront(&list, idx);
            }
        }

        // ── Helpers ──

        fn findInList(list: *const [ALPHABET_SIZE]SymbolType, sym: SymbolType) SymbolType {
            for (0..ALPHABET_SIZE) |i| {
                if (list[i] == sym) return @intCast(i);
            }
            unreachable; // all alphabet values are always in the list
        }

        fn moveToFront(list: *[ALPHABET_SIZE]SymbolType, idx: SymbolType) void {
            if (idx == 0) return;
            const val = list[idx];
            // Shift elements right by 1
            var i: u32 = idx;
            while (i > 0) : (i -= 1) {
                list[i] = list[i - 1];
            }
            list[0] = val;
        }
    };
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;
const ByteMtf = Mtf(u8);

test "MTF: basic round-trip" {
    const input = "banana";
    const mtf = try ByteMtf.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(mtf);

    const restored = try ByteMtf.inverseUniform(testing.allocator, mtf);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, input, restored);
}

test "MTF: known output for repeated 'a'" {
    const input = "aaaaaa";
    var output: [6]u8 = undefined;
    ByteMtf.forwardInPlace(input, &output);

    // First 'a' at original position 97, then subsequent 'a's are at index 0
    try testing.expectEqual(@as(u8, 97), output[0]);
    try testing.expectEqual(@as(u8, 0), output[1]);
    try testing.expectEqual(@as(u8, 0), output[2]);
    try testing.expectEqual(@as(u8, 0), output[3]);
    try testing.expectEqual(@as(u8, 0), output[4]);
    try testing.expectEqual(@as(u8, 0), output[5]);
}

test "MTF: repeated characters produce zeros" {
    const input = "xxxxxxxx";
    var output: [8]u8 = undefined;
    ByteMtf.forwardInPlace(input, &output);

    for (output[1..]) |b| {
        try testing.expectEqual(@as(u8, 0), b);
    }
}

test "MTF: alternating characters produce small values" {
    const input = "abababab";
    var output: [8]u8 = undefined;
    ByteMtf.forwardInPlace(input, &output);

    for (output[2..]) |b| {
        try testing.expectEqual(@as(u8, 1), b);
    }
}

test "MTF: in-place forward and inverse" {
    const input = "hello, world!";
    var mtf_out: [13]u8 = undefined;
    var restored: [13]u8 = undefined;

    ByteMtf.forwardInPlace(input, &mtf_out);
    ByteMtf.inverseInPlace(&mtf_out, &restored);

    try testing.expectEqualSlices(u8, input, &restored);
}

test "MTF: empty input" {
    const input = "";
    const mtf = try ByteMtf.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(mtf);
    try testing.expectEqual(@as(usize, 0), mtf.len);

    const restored = try ByteMtf.inverseUniform(testing.allocator, mtf);
    defer testing.allocator.free(restored);
    try testing.expectEqual(@as(usize, 0), restored.len);
}

test "MTF: all 256 byte values" {
    var input: [256]u8 = undefined;
    for (0..256) |i| input[i] = @intCast(i);

    const mtf = try ByteMtf.forwardUniform(testing.allocator, &input);
    defer testing.allocator.free(mtf);

    const restored = try ByteMtf.inverseUniform(testing.allocator, mtf);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, &input, restored);
}

test "MTF: BWT + MTF pipeline round-trip" {
    // Test the typical compression pipeline: BWT → MTF → inverse MTF → inverse BWT
    const Bwt = @import("bwt.zig").Bwt(u32);

    const input = "abracadabra";
    const bwt_data = try Bwt.forwardUniform(testing.allocator, input);
    defer testing.allocator.free(bwt_data);

    const bwt_bytes = Bwt.getBwtBytes(bwt_data);
    const mtf = try ByteMtf.forwardUniform(testing.allocator, bwt_bytes);
    defer testing.allocator.free(mtf);

    var small_count: usize = 0;
    for (mtf) |b| {
        if (b <= 3) small_count += 1;
    }
    try testing.expect(small_count > mtf.len / 2);

    const mtf_inv = try ByteMtf.inverseUniform(testing.allocator, mtf);
    defer testing.allocator.free(mtf_inv);
    try testing.expectEqualSlices(u8, bwt_bytes, mtf_inv);

    var reconstructed = try testing.allocator.alloc(u8, bwt_data.len);
    defer testing.allocator.free(reconstructed);
    @memcpy(reconstructed[0..4], bwt_data[0..4]);
    @memcpy(reconstructed[4..], mtf_inv);

    const restored = try Bwt.inverseUniform(testing.allocator, reconstructed);
    defer testing.allocator.free(restored);
    try testing.expectEqualSlices(u8, input, restored);
}

test "MTF: ArenaConfig variant" {
    const arena = ArenaConfig.uniform(testing.allocator);
    const input = "test_arena";

    const mtf = try ByteMtf.forward(arena, input);
    defer testing.allocator.free(mtf);

    const restored = try ByteMtf.inverse(arena, mtf);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u8, input, restored);
}

test "MTF(u4): nibble alphabet round-trip" {
    const NibbleMtf = Mtf(u4);
    const input = [_]u4{ 0xA, 0xB, 0xA, 0xC, 0xA, 0xB };

    const mtf = try NibbleMtf.forwardUniform(testing.allocator, &input);
    defer testing.allocator.free(mtf);

    const restored = try NibbleMtf.inverseUniform(testing.allocator, mtf);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u4, &input, restored);
}

test "MTF(u4): repeated nibble produces zeros after first" {
    const NibbleMtf = Mtf(u4);
    const input = [_]u4{ 5, 5, 5, 5 };
    var output: [4]u4 = undefined;
    NibbleMtf.forwardInPlace(&input, &output);

    try testing.expectEqual(@as(u4, 5), output[0]);
    try testing.expectEqual(@as(u4, 0), output[1]);
    try testing.expectEqual(@as(u4, 0), output[2]);
    try testing.expectEqual(@as(u4, 0), output[3]);
}

test "MTF(u16): word-level alphabet round-trip" {
    const WordMtf = Mtf(u16);
    const input = [_]u16{ 1000, 2000, 1000, 3000, 1000, 2000 };

    const mtf = try WordMtf.forwardUniform(testing.allocator, &input);
    defer testing.allocator.free(mtf);

    const restored = try WordMtf.inverseUniform(testing.allocator, mtf);
    defer testing.allocator.free(restored);

    try testing.expectEqualSlices(u16, &input, restored);
}
