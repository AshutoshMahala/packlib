//! tANS (Table-based Asymmetric Numeral Systems) — fast entropy coder.
//!
//! A table-based ANS implementation inspired by FSE (Finite State Entropy)
//! as used in zstandard. Builds lookup tables at encode/decode time for
//! fast per-symbol state transitions via simple table lookups.
//!
//! Compared to rANS, tANS replaces division with table lookups, making
//! encode/decode faster at the cost of a build-time table construction step.
//!
//! **Build-time:** allocates (uses `ArenaConfig`) for encoding/decoding tables.
//! **Encode/Decode:** all state transitions via prebuilt tables.
//!
//! ## Algorithm
//!
//! States live in [0, L) where L = 2^scale_bits. The "spread" function
//! distributes symbols across L positions. Each position becomes a decode
//! table entry storing (symbol, nb_bits_to_read, base_for_new_state).
//!
//! Encoding reverses decoding: given (symbol, state), emit bits and
//! transition to the decode-table position that would reconstruct this state.
//!
//! ## Serialized format
//!
//! ```text
//! [u32: final_state]         — state after encoding all symbols
//! [u32: num_output_bits]     — total bits emitted during encoding
//! [u16: scale_bits]          — log2 of table size
//! [u16: num_symbols]         — alphabet size
//! [num_symbols × u16: freqs] — quantized symbol frequencies
//! [variable: bitstream]      — packed bit output (in decode order)
//! ```
//!
//! ## References
//!
//! - Duda (2009), "Asymmetric numeral systems"
//! - Collet (2013), "Finite State Entropy" (FSE reference implementation)
//! - Giesen (2014), "Table-based ANS" (ryg_rans)

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn Tans(comptime scale_bits: u5) type {
    comptime {
        if (scale_bits < 2)
            @compileError("Tans: scale_bits must be at least 2 (table size must be >= 4)");
    }
    const L: u32 = @as(u32, 1) << scale_bits;

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

        /// Decode table entry: for each state, what symbol it decodes to
        /// and how to compute the next state.
        const DecodeEntry = struct {
            symbol: u16,
            nb_bits: u8, // bits to read from stream
            base: u32, // new_state = base + readBits(nb_bits); result in [0, L)
        };

        /// Encode table entry for a single symbol occurrence.
        /// Represents a range of states [range_start, range_start + 2^nb_bits)
        /// that all encode to the same decode-table position.
        const EncodeSlot = struct {
            range_start: u32, // inclusive start of state range
            nb_bits: u8, // bits to emit (= bits that will be read during decode)
            spread_pos: u32, // decode table position → becomes next state
        };

        /// Compiled tables.
        pub const Tables = struct {
            /// Decoding table: indexed by state [0, L)
            dec: []DecodeEntry,
            /// Per-symbol encoding slots, sorted by range_start.
            /// enc_slots[enc_offsets[s]..enc_offsets[s+1]] are the slots for symbol s.
            enc_slots: []EncodeSlot,
            enc_offsets: []u32, // length = num_symbols + 1
            freqs: []u16,
            num_symbols: u16,
        };

        // ── Quantize frequencies ──

        pub fn quantizeFreqs(
            allocator: std.mem.Allocator,
            raw_freqs: []const u32,
        ) ![]u16 {
            const num_syms: u16 = @intCast(raw_freqs.len);
            if (num_syms == 0) return Error.InvalidFrequencies;

            var total: u64 = 0;
            var active: u16 = 0;
            for (raw_freqs) |f| {
                total += f;
                if (f > 0) active += 1;
            }
            if (total == 0) return Error.InvalidFrequencies;
            if (active > L) return Error.InvalidFrequencies;

            const freqs = try allocator.alloc(u16, num_syms);

            var scaled_total: u32 = 0;
            for (0..num_syms) |i| {
                if (raw_freqs[i] == 0) {
                    freqs[i] = 0;
                } else {
                    const scaled = @max(1, @as(u32, @intCast(
                        (@as(u64, raw_freqs[i]) * L + total / 2) / total,
                    )));
                    freqs[i] = @intCast(scaled);
                    scaled_total += scaled;
                }
            }

            while (scaled_total != L) {
                if (scaled_total > L) {
                    var best: u16 = 0;
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
                    var best: u16 = 0;
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

            return freqs;
        }

        // ── Build tables ──

        pub fn buildTables(allocator: std.mem.Allocator, freqs: []const u16) !Tables {
            const num_syms: u16 = @intCast(freqs.len);

            // Verify sum
            var sum_check: u32 = 0;
            for (freqs) |f| sum_check += f;
            if (sum_check != L) return Error.FrequencySumMismatch;

            // 1. Build spread using FSE step function
            const SENTINEL: u16 = 0xFFFF;
            const spread = try allocator.alloc(u16, L);
            defer allocator.free(spread);
            @memset(spread, SENTINEL);

            const step: u32 = (L >> 1) + (L >> 3) + 3;
            const mask: u32 = L - 1;
            var pos: u32 = 0;

            for (0..num_syms) |s| {
                for (0..freqs[s]) |_| {
                    while (spread[pos] != SENTINEL) {
                        pos = (pos + 1) & mask;
                    }
                    spread[pos] = @intCast(s);
                    pos = (pos + step) & mask;
                }
            }

            // 2. Build decode table
            const dec = try allocator.alloc(DecodeEntry, L);
            errdefer allocator.free(dec);

            // Track per-symbol occurrence count
            const occurrence = try allocator.alloc(u32, num_syms);
            defer allocator.free(occurrence);
            @memset(occurrence, 0);

            for (0..L) |p| {
                const sym = spread[p];
                const f: u32 = freqs[sym];
                const k = occurrence[sym];
                occurrence[sym] = k + 1;

                // reduced state: f + k (in [f, 2f))
                const reduced: u32 = f + k;
                const nb: u8 = @intCast(@as(u32, scale_bits) - log2floor(reduced));
                const base: u32 = (reduced << @intCast(nb)) - L;

                dec[p] = .{
                    .symbol = sym,
                    .nb_bits = nb,
                    .base = base,
                };
            }

            // 3. Build encode table
            const enc_slots = try allocator.alloc(EncodeSlot, L);
            errdefer allocator.free(enc_slots);

            const enc_offsets = try allocator.alloc(u32, @as(usize, num_syms) + 1);
            errdefer allocator.free(enc_offsets);

            enc_offsets[0] = 0;
            for (0..num_syms) |s| {
                enc_offsets[s + 1] = enc_offsets[s] + freqs[s];
            }

            const sym_write_pos = try allocator.alloc(u32, num_syms);
            defer allocator.free(sym_write_pos);
            @memcpy(sym_write_pos, enc_offsets[0..num_syms]);

            for (0..L) |p| {
                const entry = dec[p];
                const sym = entry.symbol;
                const write_idx = sym_write_pos[sym];
                sym_write_pos[sym] = write_idx + 1;

                enc_slots[write_idx] = .{
                    .range_start = entry.base,
                    .nb_bits = entry.nb_bits,
                    .spread_pos = @intCast(p),
                };
            }

            // Sort each symbol's slots by range_start
            for (0..num_syms) |s| {
                const start_off = enc_offsets[s];
                const end_off = enc_offsets[s + 1];
                const slot_slice = enc_slots[start_off..end_off];
                std.sort.pdq(EncodeSlot, slot_slice, {}, struct {
                    fn cmp(_: void, a: EncodeSlot, b: EncodeSlot) bool {
                        return a.range_start < b.range_start;
                    }
                }.cmp);
            }

            // Copy freqs
            const freqs_copy = try allocator.alloc(u16, num_syms);
            errdefer allocator.free(freqs_copy);
            @memcpy(freqs_copy, freqs);

            return .{
                .dec = dec,
                .enc_slots = enc_slots,
                .enc_offsets = enc_offsets,
                .freqs = freqs_copy,
                .num_symbols = num_syms,
            };
        }

        pub fn freeTables(allocator: std.mem.Allocator, tables: *Tables) void {
            allocator.free(tables.dec);
            allocator.free(tables.enc_slots);
            allocator.free(tables.enc_offsets);
            allocator.free(tables.freqs);
        }

        // ── Encode ──

        pub fn encodeData(arena: ArenaConfig, tables: *const Tables, symbols: []const u16) ![]u8 {
            return encodeImpl(arena.temp, arena.output, tables, symbols);
        }

        pub fn encodeUniform(allocator: std.mem.Allocator, tables: *const Tables, symbols: []const u16) ![]u8 {
            return encodeImpl(allocator, allocator, tables, symbols);
        }

        fn encodeImpl(
            temp_alloc: std.mem.Allocator,
            out_alloc: std.mem.Allocator,
            tables: *const Tables,
            symbols: []const u16,
        ) ![]u8 {
            if (symbols.len == 0) return Error.EmptyInput;

            // Collect emitted bits
            var bits_buf = std.ArrayListUnmanaged(u1){};
            defer bits_buf.deinit(temp_alloc);

            // State starts at 0
            var state: u32 = 0;

            // Encode in reverse
            var i = symbols.len;
            while (i > 0) {
                i -= 1;
                const sym = symbols[i];

                // Find the encode slot for this (symbol, state) pair
                const slot = findEncodeSlot(tables, sym, state);

                // Emit nb_bits bits (the value is state - slot.range_start)
                const bits_val = state - slot.range_start;
                // Emit bits LSB-first (they get reversed during serialization,
                // so decoder reads them MSB-first as needed)
                for (0..slot.nb_bits) |b| {
                    try bits_buf.append(temp_alloc, @intCast((bits_val >> @intCast(b)) & 1));
                }

                // Transition to new state
                state = slot.spread_pos;
            }

            // Serialize
            const total_bits: u32 = @intCast(bits_buf.items.len);
            const bit_bytes: u32 = (total_bits + 7) / 8;
            const header_size: u32 = 4 + 4 + 2 + 2 + @as(u32, tables.num_symbols) * 2;
            const total_size = header_size + bit_bytes;

            const result = try out_alloc.alloc(u8, total_size);
            @memset(result, 0);

            var off: u32 = 0;
            std.mem.writeInt(u32, result[off..][0..4], state, .little);
            off += 4;
            std.mem.writeInt(u32, result[off..][0..4], total_bits, .little);
            off += 4;
            std.mem.writeInt(u16, result[off..][0..2], scale_bits, .little);
            off += 2;
            std.mem.writeInt(u16, result[off..][0..2], tables.num_symbols, .little);
            off += 2;
            for (0..tables.num_symbols) |si| {
                std.mem.writeInt(u16, result[off..][0..2], tables.freqs[si], .little);
                off += 2;
            }

            // Write bits in REVERSE order (decode reads forward, encode writes backward)
            var bit_idx: u32 = 0;
            var bi = total_bits;
            while (bi > 0) {
                bi -= 1;
                const bit_val = bits_buf.items[bi];
                const byte_pos = off + bit_idx / 8;
                const shift: u3 = @intCast(7 - (bit_idx % 8));
                result[byte_pos] |= @as(u8, bit_val) << shift;
                bit_idx += 1;
            }

            return result;
        }

        /// Binary search for the encode slot matching (symbol, state).
        fn findEncodeSlot(tables: *const Tables, sym: u16, state: u32) EncodeSlot {
            const start = tables.enc_offsets[sym];
            const end = tables.enc_offsets[sym + 1];
            const slots = tables.enc_slots[start..end];

            // Binary search: find the last slot where range_start <= state
            var lo: u32 = 0;
            var hi: u32 = @intCast(slots.len);
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (slots[mid].range_start <= state) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            // lo-1 is the slot with the largest range_start <= state
            return slots[lo - 1];
        }

        // ── Decode ──

        pub fn decodeData(arena: ArenaConfig, data: []const u8, count: u32) ![]u16 {
            return decodeImpl(arena.temp, arena.output, data, count);
        }

        pub fn decodeUniform(allocator: std.mem.Allocator, data: []const u8, count: u32) ![]u16 {
            return decodeImpl(allocator, allocator, data, count);
        }

        fn decodeImpl(
            temp_alloc: std.mem.Allocator,
            out_alloc: std.mem.Allocator,
            data: []const u8,
            count: u32,
        ) ![]u16 {
            if (data.len < 12) return Error.InvalidData;

            var off: u32 = 0;
            var state = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            const total_bits = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            const stored_scale = std.mem.readInt(u16, data[off..][0..2], .little);
            off += 2;
            if (stored_scale != scale_bits) return Error.InvalidData;
            const num_syms = std.mem.readInt(u16, data[off..][0..2], .little);
            off += 2;

            // Read frequencies
            const freqs = try temp_alloc.alloc(u16, num_syms);
            defer temp_alloc.free(freqs);
            for (0..num_syms) |si| {
                freqs[si] = std.mem.readInt(u16, data[off..][0..2], .little);
                off += 2;
            }

            // Rebuild decode table
            var tables = try buildTables(temp_alloc, freqs);
            defer freeTables(temp_alloc, &tables);

            const bit_start = off;
            var bit_pos: u32 = 0;

            // Decode forward
            const result = try out_alloc.alloc(u16, count);

            for (0..count) |di| {
                if (state >= L) return Error.DecodeError;

                const entry = tables.dec[state];
                result[di] = entry.symbol;

                // Read nb_bits from bitstream to form new state
                var bits_read: u32 = 0;
                for (0..entry.nb_bits) |_| {
                    if (bit_pos >= total_bits) return Error.DecodeError;
                    const byte_idx = bit_start + bit_pos / 8;
                    const shift: u3 = @intCast(7 - (bit_pos % 8));
                    const bit: u32 = @intCast((data[byte_idx] >> shift) & 1);
                    bits_read = (bits_read << 1) | bit;
                    bit_pos += 1;
                }

                state = entry.base + bits_read;
            }

            return result;
        }

        // ── Helpers ──

        fn log2floor(x: u32) u32 {
            if (x == 0) return 0;
            return 31 - @as(u32, @clz(x));
        }
    };
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "tANS: basic encode/decode round-trip" {
    const T = Tans(10);
    const raw_freqs = [_]u32{ 100, 50, 30, 20 };
    const freqs = try T.quantizeFreqs(testing.allocator, &raw_freqs);
    defer testing.allocator.free(freqs);

    var tables = try T.buildTables(testing.allocator, freqs);
    defer T.freeTables(testing.allocator, &tables);

    const symbols = [_]u16{ 0, 1, 2, 3, 0, 0, 1, 2 };
    const encoded = try T.encodeUniform(testing.allocator, &tables, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try T.decodeUniform(testing.allocator, encoded, @intCast(symbols.len));
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u16, &symbols, decoded);
}

test "tANS: two symbols equal frequency" {
    const T = Tans(8);
    const raw_freqs = [_]u32{ 50, 50 };
    const freqs = try T.quantizeFreqs(testing.allocator, &raw_freqs);
    defer testing.allocator.free(freqs);

    var tables = try T.buildTables(testing.allocator, freqs);
    defer T.freeTables(testing.allocator, &tables);

    const symbols = [_]u16{ 0, 1, 0, 1, 1, 0 };
    const encoded = try T.encodeUniform(testing.allocator, &tables, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try T.decodeUniform(testing.allocator, encoded, 6);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u16, &symbols, decoded);
}

test "tANS: skewed distribution" {
    const T = Tans(10);
    const raw_freqs = [_]u32{ 900, 80, 15, 5 };
    const freqs = try T.quantizeFreqs(testing.allocator, &raw_freqs);
    defer testing.allocator.free(freqs);

    var tables = try T.buildTables(testing.allocator, freqs);
    defer T.freeTables(testing.allocator, &tables);

    var symbols: [100]u16 = undefined;
    for (0..100) |i| {
        symbols[i] = if (i < 80) 0 else if (i < 90) 1 else if (i < 97) 2 else 3;
    }

    const encoded = try T.encodeUniform(testing.allocator, &tables, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try T.decodeUniform(testing.allocator, encoded, 100);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u16, &symbols, decoded);
}

test "tANS: single symbol" {
    const T = Tans(8);
    const raw_freqs = [_]u32{100};
    const freqs = try T.quantizeFreqs(testing.allocator, &raw_freqs);
    defer testing.allocator.free(freqs);

    var tables = try T.buildTables(testing.allocator, freqs);
    defer T.freeTables(testing.allocator, &tables);

    const symbols = [_]u16{ 0, 0, 0, 0, 0 };
    const encoded = try T.encodeUniform(testing.allocator, &tables, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try T.decodeUniform(testing.allocator, encoded, 5);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u16, &symbols, decoded);
}

test "tANS: frequency quantization sums to SCALE" {
    const T = Tans(10);
    const raw_freqs = [_]u32{ 70, 20, 10 };
    const freqs = try T.quantizeFreqs(testing.allocator, &raw_freqs);
    defer testing.allocator.free(freqs);

    var total: u32 = 0;
    for (freqs) |f| total += f;
    try testing.expectEqual(@as(u32, 1024), total);
}

test "tANS: ArenaConfig variant" {
    const T = Tans(8);
    const arena = ArenaConfig.uniform(testing.allocator);

    const raw_freqs = [_]u32{ 60, 30, 10 };
    const freqs = try T.quantizeFreqs(testing.allocator, &raw_freqs);
    defer testing.allocator.free(freqs);

    var tables = try T.buildTables(testing.allocator, freqs);
    defer T.freeTables(testing.allocator, &tables);

    const symbols = [_]u16{ 0, 1, 2, 0, 0, 1 };
    const encoded = try T.encodeData(arena, &tables, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try T.decodeData(arena, encoded, 6);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u16, &symbols, decoded);
}

test "tANS: larger alphabet (10 symbols)" {
    const T = Tans(10);
    var raw_freqs: [10]u32 = undefined;
    for (0..10) |i| raw_freqs[i] = @intCast(10 + (10 - i) * 5);

    const freqs = try T.quantizeFreqs(testing.allocator, &raw_freqs);
    defer testing.allocator.free(freqs);

    var tables = try T.buildTables(testing.allocator, freqs);
    defer T.freeTables(testing.allocator, &tables);

    var symbols: [50]u16 = undefined;
    for (0..50) |i| symbols[i] = @intCast(i % 10);

    const encoded = try T.encodeUniform(testing.allocator, &tables, &symbols);
    defer testing.allocator.free(encoded);

    const decoded = try T.decodeUniform(testing.allocator, encoded, 50);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u16, &symbols, decoded);
}
