//! Comptime Utilities — shared comptime helpers for packlib modules.
//!
//! These functions return types or values computed at compile time.
//! They are used to derive optimal type parameters for generic modules
//! like `Dafsa`, `WaveletTree`, and `Bpe`.

const std = @import("std");

/// Given a slice of symbol sequences, returns the smallest unsigned integer
/// type that can represent every symbol value present.
///
/// Uses the exact bit width needed — e.g. `u7` for ASCII lowercase ('z' = 122),
/// `u14` for BMP CJK, `u21` for full Unicode. Zig supports arbitrary-width
/// integers so there is no rounding to power-of-two widths.
///
/// Usage:
/// ```
/// const syms: []const []const u21 = &.{ &.{'a','b'}, &.{'z'} };
/// const T = detectSymbolType(u21, syms);
/// // T = u7 (max value 'z' = 122, fits in 7 bits)
/// const D = packlib.Dafsa(T, u32);
/// ```
///
/// Both `InputSymbol` and `sequences` must be comptime-known. This is inherent
/// to the design: the result is a *type*, which only exists at comptime in Zig.
/// A runtime equivalent is not possible because Zig types are erased at runtime.
pub fn detectSymbolType(comptime InputSymbol: type, comptime sequences: []const []const InputSymbol) type {
    comptime {
        var max_val: InputSymbol = 0;
        for (sequences) |seq| {
            for (seq) |sym| {
                if (sym > max_val) max_val = sym;
            }
        }
        if (max_val == 0) return u1;
        const bits_needed: usize = @as(usize, std.math.log2_int(InputSymbol, max_val)) + 1;
        return std.meta.Int(.unsigned, bits_needed);
    }
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "detectSymbolType: ASCII lowercase" {
    const seqs = [_][]const u21{ &.{ 'a', 'b', 'c' }, &.{ 'x', 'y', 'z' } };
    const T = detectSymbolType(u21, &seqs);
    // max = 'z' = 122, needs 7 bits exactly
    try testing.expect(T == u7);
}

test "detectSymbolType: small values" {
    const seqs = [_][]const u16{ &.{ 0, 1, 2 }, &.{ 3, 7 } };
    const T = detectSymbolType(u16, &seqs);
    // max = 7, needs 3 bits exactly
    try testing.expect(T == u3);
}

test "detectSymbolType: single zero" {
    const seqs = [_][]const u8{&.{0}};
    const T = detectSymbolType(u8, &seqs);
    try testing.expect(T == u1);
}

test "detectSymbolType: Unicode codepoints" {
    // Japanese Katakana ア (U+30A2) = 12450
    const seqs = [_][]const u21{ &.{ 0x30A2, 0x30AB }, &.{0x30B5} };
    const T = detectSymbolType(u21, &seqs);
    // max = 0x30B5 = 12469, needs 14 bits exactly
    try testing.expect(T == u14);
}

test "detectSymbolType: full byte range" {
    const seqs = [_][]const u8{&.{ 0, 255 }};
    const T = detectSymbolType(u8, &seqs);
    // max = 255, needs 8 bits → u8
    try testing.expect(T == u8);
}
