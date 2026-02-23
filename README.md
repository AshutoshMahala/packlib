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

### Tier 1 — Core Algorithms

| Module | Description | Depends on |
|--------|-------------|------------|
| `RankSelect` | O(1) rank, O(log n) select on bit vectors. | BitReader, BitWriter |
| `HuffmanCodec` | Static canonical Huffman coding. | BitReader, BitWriter, FixedPoint |
| `DeltaVarint` | Delta + LEB128 varint for sorted integer sequences. | BitReader, BitWriter |
| `Bpe` | Byte-pair encoding for alphabet reduction. | BitReader, BitWriter |
| `Entropy` | Shannon entropy analysis (offline diagnostic). | — |

### Tier 2+ — (Planned)

LOUDS trees, Elias-Fano, DAFSA, BWT, front coding, context models, ANS.

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

// Bit-level I/O
var reader = packlib.BitReader.init(data);
const val = try reader.readBits(u12, 12);

// Fixed-point math
const half = packlib.Q16_16.fromFraction(1, 2);
const quarter = half.mul(half);
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
│   ├── encoding/
│   │   ├── huffman.zig
│   │   ├── delta_varint.zig
│   │   └── bpe.zig
│   └── succinct/
│       └── rank_select.zig
└── tests/
    ├── integration.zig       # Cross-module round-trip tests
    └── test_vectors.zig      # Reference tests from published standards
```

## Status

Tier 0 and Tier 1 are fully implemented and tested. Tier 2 (LOUDS, Elias-Fano, DAFSA, BWT, etc.) is planned.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
