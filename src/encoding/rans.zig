//! rANS (Range Asymmetric Numeral Systems) — entropy coder.
//!
//! An entropy coder based on ANS (Asymmetric Numeral Systems) by Jarek Duda.
//! rANS encodes each symbol by splitting the state into a quotient and
//! remainder, achieving near-entropy compression with a simple table lookup.
//!
//! This implementation uses 32-bit state with 16-bit renormalization.
//! Symbols are encoded in reverse order and decoded forward (stack-like).
//!
//! **Build-time:** allocates (uses `ArenaConfig`).
//! **Decode-time:** zero allocation given the encoded data and frequency table.
//!
//! ## Algorithm
//!
//! Encoding (for each symbol, in reverse order):
//!   x' = (x / freq[s]) << scale_bits + cumul[s] + (x % freq[s])
//!   Renormalize: while x' >= (1 << 16) << scale_bits, output low 16 bits.
//!
//! Decoding (forward):
//!   slot = x & ((1 << scale_bits) - 1)
//!   Find symbol s such that cumul[s] <= slot < cumul[s] + freq[s]
//!   x' = freq[s] * (x >> scale_bits) + slot - cumul[s]
//!   Renormalize: while x' < (1 << 16), read 16 bits into state.
//!
//! ## Generic parameters
//!
//!   - `scale_bits` — log₂ of probability resolution (typical: 10–16)
//!   - `SymbolType` — alphabet width: `u4` (nibble), `u8` (byte), `u12`/`u16` (token)
//!
//! Symbol values are typed `SymbolType`; `num_symbols` is the active alphabet
//! size and may be ≤ `1 << @bitSizeOf(SymbolType)`.
//!
//! ## Serialized format
//!
//! ```text
//! [u32: final_state]         — ANS state after encoding all symbols
//! [u16: num_output_words]    — number of 16-bit renormalization words
//! [u16: scale_bits]          — log2 of probability resolution
//! [u32: num_symbols]         — number of symbols in the alphabet
//! [num_symbols × u16: freqs] — quantized frequency for each symbol
//! [variable: u16 words]      — renormalization output (in decode order)
//! ```
//!
//! ## References
//!
//! - Duda (2009), "Asymmetric numeral systems"
//! - Duda (2013), "Asymmetric numeral systems: entropy coding combining speed of Huffman..."
//! - Giesen (2014), "Interleaved entropy coders" (practical rANS tutorial)

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn Rans(comptime scale_bits: u5, comptime SymbolType: type) type {
    comptime {
        if (scale_bits < 2)
            @compileError("Rans: scale_bits must be at least 2 (SCALE must be >= 4)");
        const info = @typeInfo(SymbolType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("Rans: SymbolType must be an unsigned integer type, got " ++ @typeName(SymbolType));
        const bits = @bitSizeOf(SymbolType);
        if (bits == 0)
            @compileError("Rans: SymbolType must have at least 1 bit");
        if (bits > 16)
            @compileError("Rans: SymbolType must be at most 16 bits (alphabet up to 65 536)");
    }
    const SCALE: u32 = @as(u32, 1) << scale_bits;
    const RANS_L: u32 = 1 << 16; // lower bound on state
    const MAX_ALPHABET: u32 = 1 << @bitSizeOf(SymbolType);

    return struct {
        const Self = @This();

        pub const Error = error{
            EmptyInput,
            InvalidFrequencies,
            FrequencySumMismatch,
            OutOfMemory,
            InvalidData,
            DecodeError,
        };

        /// Frequency table for encoding/decoding.
        pub const FreqTable = struct {
            freqs: []u16, // quantized frequency per symbol
            cumul: []u32, // cumulative frequencies (length = num_symbols + 1)
            num_symbols: u32,

            /// Verify that frequencies sum to SCALE.
            pub fn validate(self: *const FreqTable) bool {
                return self.cumul[self.num_symbols] == SCALE;
            }
        };

        // ── Build frequency table ──

        /// Build a quantized frequency table from raw counts.
        /// Scales frequencies to sum to exactly SCALE, ensuring no symbol
        /// with nonzero count gets zero frequency.
        pub fn buildFreqTable(
            allocator: std.mem.Allocator,
            raw_freqs: []const u32,
        ) !FreqTable {
            if (raw_freqs.len == 0) return Error.InvalidFrequencies;
            if (raw_freqs.len > MAX_ALPHABET) return Error.InvalidFrequencies;
            const num_syms: u32 = @intCast(raw_freqs.len);

            var total: u64 = 0;
            var active: u32 = 0;
            for (raw_freqs) |f| {
                total += f;
                if (f > 0) active += 1;
            }
            if (total == 0) return Error.InvalidFrequencies;
            if (active > SCALE) return Error.InvalidFrequencies;

            const freqs = try allocator.alloc(u16, num_syms);
            errdefer allocator.free(freqs);
            const cumul = try allocator.alloc(u32, @as(usize, num_syms) + 1);
            errdefer allocator.free(cumul);

            // Initial scaling
            var scaled_total: u32 = 0;
            for (0..num_syms) |i| {
                if (raw_freqs[i] == 0) {
                    freqs[i] = 0;
                } else {
                    // Scale proportionally, minimum 1
                    const scaled = @max(1, @as(u32, @intCast((@as(u64, raw_freqs[i]) * SCALE + total / 2) / total)));
                    freqs[i] = @intCast(scaled);
                    scaled_total += scaled;
                }
            }

            // Adjust to sum exactly to SCALE
            while (scaled_total != SCALE) {
                if (scaled_total > SCALE) {
                    // Find symbol with largest freq > 1 and decrement
                    var best: u32 = 0;
                    var best_freq: u16 = 0;
                    for (0..num_syms) |i| {
                        if (freqs[i] > 1 and freqs[i] > best_freq) {
                            best = @intCast(i);
                            best_freq = freqs[i];
                        }
                    }
                    freqs[best] -= 1;
                    scaled_total -= 1;
                } else {
                    // Find symbol with largest raw freq and increment
                    var best: u32 = 0;
                    var best_raw: u32 = 0;
                    for (0..num_syms) |i| {
                        if (raw_freqs[i] > best_raw) {
                            best = @intCast(i);
                            best_raw = raw_freqs[i];
                        }
                    }
                    freqs[best] += 1;
                    scaled_total += 1;
                }
            }

            // Build cumulative
            cumul[0] = 0;
            for (0..num_syms) |i| {
                cumul[i + 1] = cumul[i] + freqs[i];
            }

            return .{
                .freqs = freqs,
                .cumul = cumul,
                .num_symbols = num_syms,
            };
        }

        pub fn freeFreqTable(allocator: std.mem.Allocator, table: *FreqTable) void {
            allocator.free(table.freqs);
            allocator.free(table.cumul);
        }

        // ── Encode ──

        /// Encode a sequence of symbol indices using rANS.
        /// `symbols` contains symbol indices (must be < num_symbols).
        pub fn encode(arena: ArenaConfig, table: *const FreqTable, symbols: []const SymbolType) ![]u8 {
            return encodeImpl(arena.temp, arena.output, table, symbols);
        }

        pub fn encodeUniform(allocator: std.mem.Allocator, table: *const FreqTable, symbols: []const SymbolType) ![]u8 {
            return encodeImpl(allocator, allocator, table, symbols);
        }

        fn encodeImpl(
            temp: std.mem.Allocator,
            output: std.mem.Allocator,
            table: *const FreqTable,
            symbols: []const SymbolType,
        ) ![]u8 {
            if (symbols.len == 0) return Error.EmptyInput;

            // Encode in reverse order (rANS is LIFO)
            var state: u32 = RANS_L; // initial state

            // Collect renormalization words
            var words = std.ArrayListUnmanaged(u16){};
            defer words.deinit(temp);

            var i = symbols.len;
            while (i > 0) {
                i -= 1;
                const s = symbols[i];
                const freq = table.freqs[s];
                const start = table.cumul[s];

                // Renormalize: ensure state doesn't overflow after encoding
                // upper = ((RANS_L >> scale_bits) * freq) << 16 (may exceed u32)
                const upper: u64 = @as(u64, RANS_L >> scale_bits) * @as(u64, freq) << 16;
                while (@as(u64, state) >= upper) {
                    try words.append(temp, @intCast(state & 0xFFFF));
                    state >>= 16;
                }

                // Encode: x' = (x / f) << scale_bits + cumul + (x % f)
                state = (state / freq) * SCALE + @as(u32, @intCast(start)) + (state % freq);
            }

            // Serialize
            const num_words: u32 = @intCast(words.items.len);
            // Header: state(4) + num_words(2) + scale_bits(2) + num_syms(4) + freqs(num_syms*2) + words(num_words*2)
            const header_size: u32 = 4 + 2 + 2 + 4 + table.num_symbols * 2;
            const total_size = header_size + num_words * 2;
            const result = try output.alloc(u8, total_size);

            var off: u32 = 0;
            std.mem.writeInt(u32, result[off..][0..4], state, .little);
            off += 4;
            std.mem.writeInt(u16, result[off..][0..2], @intCast(num_words), .little);
            off += 2;
            std.mem.writeInt(u16, result[off..][0..2], scale_bits, .little);
            off += 2;
            std.mem.writeInt(u32, result[off..][0..4], table.num_symbols, .little);
            off += 4;

            // Write frequencies
            for (0..table.num_symbols) |si| {
                std.mem.writeInt(u16, result[off..][0..2], table.freqs[si], .little);
                off += 2;
            }

            // Write renormalization words in REVERSE order (so they decode forward)
            var wi = num_words;
            while (wi > 0) {
                wi -= 1;
                std.mem.writeInt(u16, result[off..][0..2], words.items[wi], .little);
                off += 2;
            }

            return result;
        }

        // ── Decode ──

        /// Decode rANS-encoded data back to symbol indices.
        pub fn decode(arena: ArenaConfig, data: []const u8, count: u32) ![]SymbolType {
            return decodeImpl(arena.temp, arena.output, data, count);
        }

        pub fn decodeUniform(allocator: std.mem.Allocator, data: []const u8, count: u32) ![]SymbolType {
            return decodeImpl(allocator, allocator, data, count);
        }

        fn decodeImpl(
            temp: std.mem.Allocator,
            output_alloc: std.mem.Allocator,
            data: []const u8,
            count: u32,
        ) ![]SymbolType {
            if (data.len < 12) return Error.InvalidData;

            var off: u32 = 0;
            var state = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            // skip num_words (u16) — we read words until state is renormalized
            off += 2;
            const stored_scale = std.mem.readInt(u16, data[off..][0..2], .little);
            off += 2;
            if (stored_scale != scale_bits) return Error.InvalidData;

            const num_syms = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            if (num_syms > MAX_ALPHABET) return Error.InvalidData;

            // Read frequencies and build cumulative
            const freqs = try temp.alloc(u16, num_syms);
            defer temp.free(freqs);
            const cumul = try temp.alloc(u32, @as(usize, num_syms) + 1);
            defer temp.free(cumul);

            cumul[0] = 0;
            for (0..num_syms) |i| {
                freqs[i] = std.mem.readInt(u16, data[off..][0..2], .little);
                off += 2;
                cumul[i + 1] = cumul[i] + freqs[i];
            }

            // Build symbol lookup table for fast decoding
            const sym_table = try temp.alloc(SymbolType, SCALE);
            defer temp.free(sym_table);
            for (0..num_syms) |s| {
                const start = cumul[s];
                const end = cumul[s + 1];
                for (start..end) |j| {
                    sym_table[j] = @intCast(s);
                }
            }

            // Read renormalization words
            var word_pos: u32 = off;

            // Decode
            const result = try output_alloc.alloc(SymbolType, count);
            for (0..count) |i| {
                const slot = state & (SCALE - 1);
                const s = sym_table[slot];
                result[i] = s;

                // Advance state
                const freq = freqs[s];
                const start = cumul[s];
                state = @as(u32, freq) * (state >> scale_bits) + slot - start;

                // Renormalize
                while (state < RANS_L) {
                    if (word_pos + 2 > data.len) return Error.DecodeError;
                    const word = std.mem.readInt(u16, data[word_pos..][0..2], .little);
                    state = (state << 16) | word;
                    word_pos += 2;
                }
            }

            return result;
        }
    };
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "rANS: basic encode/decode round-trip" {
    const R = Rans(12, u8); // 12-bit scale = 4096, byte alphabet

    // Simple symbol frequencies
    const raw_freqs = [_]u32{ 100, 50, 30, 20 };
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    try testing.expect(table.validate());

    // Encode some symbols
    const symbols = [_]u8{ 0, 1, 2, 3, 0, 0, 1, 2 };
    const encoded = try R.encodeUniform(testing.allocator, &table, &symbols);
    defer testing.allocator.free(encoded);

    // Decode
    const decoded = try R.decodeUniform(testing.allocator, encoded, @intCast(symbols.len));
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, &symbols, decoded);
}

test "rANS: single symbol" {
    const R = Rans(10, u8);
    const raw_freqs = [_]u32{100};
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    const symbols = [_]u8{ 0, 0, 0, 0, 0 };
    const encoded = try R.encodeUniform(testing.allocator, &table, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try R.decodeUniform(testing.allocator, encoded, 5);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, &symbols, decoded);
}

test "rANS: two symbols equal frequency" {
    const R = Rans(10, u8);
    const raw_freqs = [_]u32{ 50, 50 };
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    const symbols = [_]u8{ 0, 1, 0, 1, 0, 1 };
    const encoded = try R.encodeUniform(testing.allocator, &table, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try R.decodeUniform(testing.allocator, encoded, 6);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, &symbols, decoded);
}

test "rANS: skewed distribution" {
    const R = Rans(12, u8);
    const raw_freqs = [_]u32{ 1000, 10, 5, 1 };
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    // Generate a sequence
    var symbols: [100]u8 = undefined;
    for (0..100) |i| {
        symbols[i] = if (i < 80) 0 else if (i < 90) 1 else if (i < 97) 2 else 3;
    }

    const encoded = try R.encodeUniform(testing.allocator, &table, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try R.decodeUniform(testing.allocator, encoded, 100);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, &symbols, decoded);
}

test "rANS: frequency table validation" {
    const R = Rans(10, u8);
    const raw_freqs = [_]u32{ 70, 20, 10 };
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    // Frequencies should sum to 2^10 = 1024
    try testing.expect(table.validate());
    try testing.expectEqual(@as(u32, 1024), table.cumul[table.num_symbols]);
}

test "rANS: larger alphabet (26 symbols)" {
    const R = Rans(12, u8);
    var raw_freqs: [26]u32 = undefined;
    for (0..26) |i| raw_freqs[i] = @intCast(26 - i); // decreasing
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    var symbols: [200]u8 = undefined;
    for (0..200) |i| symbols[i] = @intCast(i % 26);

    const encoded = try R.encodeUniform(testing.allocator, &table, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try R.decodeUniform(testing.allocator, encoded, 200);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, &symbols, decoded);
}

test "rANS: ArenaConfig variant" {
    const R = Rans(10, u8);
    const arena = ArenaConfig.uniform(testing.allocator);

    const raw_freqs = [_]u32{ 60, 30, 10 };
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    const symbols = [_]u8{ 0, 1, 2, 0, 0, 1 };
    const encoded = try R.encode(arena, &table, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try R.decode(arena, encoded, 6);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, &symbols, decoded);
}

test "rANS: compression ratio for skewed data" {
    const R = Rans(12, u8);
    // Very skewed: symbol 0 appears 95%
    const raw_freqs = [_]u32{ 950, 30, 15, 5 };
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);

    var symbols: [1000]u8 = undefined;
    for (0..1000) |i| {
        symbols[i] = if (i < 950) 0 else if (i < 980) 1 else if (i < 995) 2 else 3;
    }

    const encoded = try R.encodeUniform(testing.allocator, &table, &symbols);
    defer testing.allocator.free(encoded);

    // ANS should compress well: 1000 symbols with H ≈ 0.35 bits/sym
    // So ~350 bits ≈ 44 bytes of payload, plus overhead
    try testing.expect(encoded.len < symbols.len); // Must be smaller than raw
}

test "rANS(12, u4): nibble alphabet round-trip" {
    const R = Rans(12, u4);
    const raw_freqs = [_]u32{ 200, 100, 50, 25, 10, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0 };
    var table = try R.buildFreqTable(testing.allocator, &raw_freqs);
    defer R.freeFreqTable(testing.allocator, &table);
    try testing.expect(table.validate());

    const symbols = [_]u4{ 0, 1, 2, 3, 0, 0, 1, 4, 0, 0, 5, 0 };
    const encoded = try R.encodeUniform(testing.allocator, &table, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try R.decodeUniform(testing.allocator, encoded, @intCast(symbols.len));
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u4, &symbols, decoded);
}
