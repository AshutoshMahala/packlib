# Packlib

Compression and succinct data structure primitives for Zig.

Packlib provides a composable toolkit of bit-level I/O, entropy coding, integer compression, and succinct data structures. All read-path operations are zero-allocation — they operate on borrowed byte slices with no heap usage.

## Design Principles

- **Zero-allocation reads.** Every query and decode function takes `[]const u8` and returns values without touching an allocator. Build/encode paths allocate as needed.
- **Generic over index and value types.** Structures are parameterized by `u16`, `u32`, or `u64` so users pay only for the range they need.
- **Deterministic.** Fixed-point arithmetic and integer-only code paths guarantee identical results across platforms and compilations.
- **Composable.** Modules are independent building blocks. Combine a `BitWriter` with `HuffmanCodec` and `RankSelect` however you like — there's no framework to buy into.

## Modules

### Tier 0 — Foundational Primitives

| Module | Description |
|--------|-------------|
| `BitReader` | Read-only bit-level cursor over `[]const u8`. MSB-first. |
| `BitWriter` | Allocator-backed bit-level output buffer. MSB-first. |
| `FixedPoint` | Generic fixed-point arithmetic (`Q16_16`, `Q32_32`). Deterministic. |

### Memory Management

| Module | Description |
|--------|-------------|
| `ArenaConfig` | Three-tier arena allocator config (temp, intermediate, output). |

### Tier 1 — Core Algorithms

| Module | Description | Depends on |
|--------|-------------|------------|
| `RankSelect` | O(1) rank, O(log n) select on bit vectors. | BitReader, BitWriter |
| `HuffmanCodec` | Static canonical Huffman coding. | BitReader, BitWriter, FixedPoint |
| `DeltaVarint` | Delta + LEB128 varint for sorted integer sequences. | BitReader, BitWriter |
| `Bpe` | Byte-pair encoding for alphabet reduction. | BitReader, BitWriter |
| `Entropy` | Shannon entropy analysis (offline diagnostic). | — |

### Tier 2 — Advanced Algorithms

| Module | Description | Depends on |
|--------|-------------|------------|
| `EliasFano` | Quasi-succinct monotone integer sequence encoding. | RankSelect, BitReader, BitWriter |
| `Louds` | Level-order unary degree sequence tree encoding. | RankSelect, BitReader, BitWriter |
| `WaveletTree` | Rank/select/access on arbitrary alphabets. | RankSelect, BitReader, BitWriter |
| `Dafsa` | Directed acyclic finite state automaton (compact string set). | — |
| `Bwt` | Burrows-Wheeler transform via suffix array. | — |
| `Mtf` | Move-to-front transform. | — |
| `FrontCoding` | Prefix-compressed sorted string storage. | — |
| `Rans` | Range asymmetric numeral systems (rANS) entropy coder. | BitReader, BitWriter |
| `Tans` | Table-based ANS (tANS / FSE-style) entropy coder. | BitReader, BitWriter |
| `ContextModel0` | Order-0 byte context model (Q16.16 fixed-point). | FixedPoint |
| `ContextModel1` | Order-1 byte context model (Q16.16 fixed-point). | FixedPoint |
| `ContextModelN` | Order-N byte context model (Q16.16 fixed-point). | FixedPoint |

## Usage

Add packlib as a Zig dependency:

```zig
// build.zig.zon
.dependencies = .{
    .packlib = .{
        .url = "...",
        .hash = "...",
    },
},
```

```zig
// build.zig
const packlib_dep = b.dependency("packlib", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("packlib", packlib_dep.module("packlib"));
```

Then in your code:

```zig
const packlib = @import("packlib");
const std = @import("std");

// Bit-level I/O
var reader = packlib.BitReader.init(data);
const val = try reader.readBits(u12, 12);

// Fixed-point math
const half = packlib.Q16_16.fromFraction(1, 2);
const quarter = half.mul(half);

// Elias-Fano: compact sorted integer sequences
const EF = packlib.EliasFano(u32);
const ef_data = try EF.buildUniform(allocator, &[_]u32{ 2, 3, 5, 7, 11, 13 });
const val = EF.get(ef_data, 3); // → 7

// Burrows-Wheeler + Move-to-Front pipeline
const Bwt = packlib.Bwt(u32);
const bwt_data = try Bwt.forwardUniform(allocator, "banana");
const bwt_bytes = Bwt.getBwtBytes(bwt_data); // → "nnbaaa"
const mtf_out = try packlib.Mtf.forwardUniform(allocator, bwt_bytes);

// rANS entropy coding
const R = packlib.Rans(12);
var table = try R.buildFreqTable(allocator, &raw_freqs);
const encoded = try R.encodeUniform(allocator, &table, symbols);
const decoded = try R.decodeUniform(allocator, encoded, symbol_count);

// LOUDS tree
const L = packlib.Louds(u32);
const tree = try L.buildFromDegreesUniform(allocator, &degrees);
const child = L.firstChild(tree, 0);
const par = L.parent(tree, child);
```

## Building & Testing

```sh
# Build
zig build

# Run all tests
zig build test

# Unit tests only
zig build test-unit

# Integration tests only
zig build test-integration

# Reference test vectors (Wikipedia / published standards)
zig build test-vectors
```

## Project Structure

```
packlib/
├── build.zig
├── build.zig.zon
├── README.md
├── LICENSE-MIT
├── LICENSE-APACHE
├── algorithm_requirements.md
├── src/
│   ├── root.zig              # Public API — re-exports all modules
│   ├── io/
│   │   ├── bit_reader.zig
│   │   └── bit_writer.zig
│   ├── math/
│   │   ├── fixed_point.zig
│   │   └── entropy.zig
│   ├── memory/
│   │   └── arena_config.zig
│   ├── encoding/
│   │   ├── huffman.zig
│   │   ├── delta_varint.zig
│   │   ├── bpe.zig
│   │   ├── bwt.zig
│   │   ├── mtf.zig
│   │   ├── front_coding.zig
│   │   ├── context_model.zig
│   │   ├── rans.zig
│   │   └── tans.zig
│   └── succinct/
│       ├── rank_select.zig
│       ├── elias_fano.zig
│       ├── louds.zig
│       ├── wavelet_tree.zig
│       └── dafsa.zig
└── tests/
    ├── integration.zig       # Cross-module round-trip tests
    └── test_vectors.zig      # Reference tests from published standards
```

## Status

Tiers 0, 1, and 2 are fully implemented and tested — 206+ unit tests, integration tests, and reference test vectors from published standards (Shannon, Vigna, Jacobson, Duda, Daciuk, etc.).

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
