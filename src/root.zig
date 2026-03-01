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

// ── Memory Management ────────────────────────────────────────
pub const ArenaConfig = @import("memory/arena_config.zig").ArenaConfig;

// ── Tier 1: Core Algorithms ─────────────────────────────────
pub const RankSelect = @import("succinct/rank_select.zig").RankSelect;
pub const HuffmanCodec = @import("encoding/huffman.zig").HuffmanCodec;
pub const DeltaVarint = @import("encoding/delta_varint.zig").DeltaVarint;
pub const Bpe = @import("encoding/bpe.zig").Bpe;
pub const Entropy = @import("math/entropy.zig").Entropy;

// ── Tier 2: Advanced Algorithms ─────────────────────────────
pub const EliasFano = @import("succinct/elias_fano.zig").EliasFano;
pub const Louds = @import("succinct/louds.zig").Louds;
pub const Bwt = @import("encoding/bwt.zig").Bwt;
pub const Mtf = @import("encoding/mtf.zig").Mtf;
pub const FrontCoding = @import("encoding/front_coding.zig").FrontCoding;
pub const WaveletTree = @import("succinct/wavelet_tree.zig").WaveletTree;
pub const ContextModel0 = @import("encoding/context_model.zig").ContextModel0;
pub const ContextModel1 = @import("encoding/context_model.zig").ContextModel1;
pub const ContextModelN = @import("encoding/context_model.zig").ContextModelN;
pub const Rans = @import("encoding/rans.zig").Rans;
pub const Tans = @import("encoding/tans.zig").Tans;
pub const Dafsa = @import("succinct/dafsa.zig").Dafsa;

// ── Pull in all module tests ─────────────────────────────────
comptime {
    _ = @import("io/bit_reader.zig");
    _ = @import("io/bit_writer.zig");
    _ = @import("math/fixed_point.zig");
    _ = @import("memory/arena_config.zig");
    _ = @import("succinct/rank_select.zig");
    _ = @import("encoding/huffman.zig");
    _ = @import("encoding/delta_varint.zig");
    _ = @import("encoding/bpe.zig");
    _ = @import("math/entropy.zig");
    _ = @import("succinct/elias_fano.zig");
    _ = @import("succinct/louds.zig");
    _ = @import("encoding/bwt.zig");
    _ = @import("encoding/mtf.zig");
    _ = @import("encoding/front_coding.zig");
    _ = @import("succinct/wavelet_tree.zig");
    _ = @import("encoding/context_model.zig");
    _ = @import("encoding/rans.zig");
    _ = @import("encoding/tans.zig");
    _ = @import("succinct/dafsa.zig");
}
