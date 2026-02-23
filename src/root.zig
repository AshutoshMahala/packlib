//! Packlib — Compression & Succinct Data Structure Primitives
//!
//! A zero-allocation (read path), generic library providing:
//! - Bit-level I/O primitives
//! - Fixed-point deterministic arithmetic
//! - Succinct data structures (rank/select, LOUDS, Elias-Fano)
//! - Entropy coding (Huffman, ANS)
//! - Preprocessing transforms (BPE, BWT, delta-varint)
//!
//! Usage:
//!   const packlib = @import("packlib");
//!   var reader = packlib.BitReader.init(data);

// ── Tier 0: Foundational Primitives ──────────────────────────
pub const BitReader = @import("io/bit_reader.zig").BitReader;
pub const BitWriter = @import("io/bit_writer.zig").BitWriter;
pub const FixedPoint = @import("math/fixed_point.zig").FixedPoint;
pub const Q16_16 = @import("math/fixed_point.zig").Q16_16;
pub const Q32_32 = @import("math/fixed_point.zig").Q32_32;

// ── Tier 1: Core Algorithms ─────────────────────────────────
pub const RankSelect = @import("succinct/rank_select.zig").RankSelect;
pub const HuffmanCodec = @import("encoding/huffman.zig").HuffmanCodec;
pub const DeltaVarint = @import("encoding/delta_varint.zig").DeltaVarint;
pub const Bpe = @import("encoding/bpe.zig").Bpe;
pub const Entropy = @import("math/entropy.zig").Entropy;

// ── Pull in all module tests ─────────────────────────────────
comptime {
    _ = @import("io/bit_reader.zig");
    _ = @import("io/bit_writer.zig");
    _ = @import("math/fixed_point.zig");
    _ = @import("succinct/rank_select.zig");
    _ = @import("encoding/huffman.zig");
    _ = @import("encoding/delta_varint.zig");
    _ = @import("encoding/bpe.zig");
    _ = @import("math/entropy.zig");
}
