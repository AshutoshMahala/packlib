//! DAFSA (Directed Acyclic Finite State Automaton) — compact string/symbol-sequence set.
//!
//! `Dafsa(SymbolType, IndexType)` is generic over:
//!  - `SymbolType` — the label/alphabet type on edges:
//!    - `u8` for byte strings (most common)
//!    - `u16`/`u21` for Unicode codepoints
//!    - `u16`/`u32` for token-level automata (e.g. BPE token IDs)
//!  - `IndexType` — the state/edge index width:
//!    - `u16` for small automata (< 65K states, halves pointer overhead)
//!    - `u32` for large automata (default, up to ~4B states)
//!    - `u64` for very large automata
//!
//! A DAFSA (also called a DAWG — Directed Acyclic Word Graph) compresses
//! a sorted set of symbol sequences by sharing both prefixes (like a trie) and
//! suffixes. This produces a minimal-state automaton that recognizes
//! exactly the input set.
//!
//! **Build-time:** allocates (uses `ArenaConfig`).
//! **Query-time:** zero allocation — lookup on `[]const u8`.
//!
//! ## Construction
//!
//! Uses the incremental algorithm of Daciuk et al. (2000):
//! 1. Process input sequences in sorted order.
//! 2. Maintain a "register" of unique right-language states.
//! 3. When a state's right-language is frozen (all its descendants are
//!    finalized), check the register for an equivalent frozen state.
//!    If found, redirect the edge to the register entry.
//!
//! ## Serialized format
//!
//! ```text
//! [IndexType: num_states]                                  — total states
//! [IndexType: num_edges]                                   — total edges
//! [num_states × IndexType: edge_offset]                    — offset into edge array for each state
//! [num_states × u8: is_final]                              — 1 if state is accepting, 0 otherwise
//! [num_edges × (SymbolType + IndexType): edges]            — (label, target_state) per edge
//! ```
//!
//! Root is always state 0 (the first state appended during construction).
//! Both standard and compact formats share this invariant.
//!
//! ## Compact serialized format
//!
//! ```text
//! [u8: magic (0xDA)]
//! [IndexType: num_states]
//! [IndexType: num_edges]
//! [num_states × DegreeType: degree per state]
//! [ceil(num_states/8) bytes: finals bitvector]
//! [num_edges × (SymbolType + IndexType): edges]            — (label, target_state) per edge
//! ```
//!
//! `DegreeType` is derived from `SymbolType`: the smallest standard unsigned
//! integer wide enough to hold `0..2^symbol_bits` (e.g. `u16` for `u8` symbols,
//! `u32` for `u21` Unicode codepoints).
//!
//! **Tradeoff:** The compact format saves space (bitvector finals, degree-per-state
//! instead of IndexType-sized offset table) but query lookups are O(state_index)
//! per edge because edge offsets are computed by summing degrees. The standard
//! format has O(1) edge offset lookup. For large corpora (100K+ states), prefer
//! the standard format when query latency matters. Use compact when minimizing
//! serialized size is the priority (e.g. embedded, shipping dictionaries as assets).
//!
//! ## References
//!
//! - Daciuk et al. (2000), "Incremental Construction of Minimal Acyclic
//!   Finite State Automata"
//! - Lucchesi and Kowaltowski (1993), "Applications of Finite Automata
//!   Representing Large Vocabularies"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub fn Dafsa(comptime SymbolType: type, comptime IndexType: type) type {
    comptime {
        const sym_info = @typeInfo(SymbolType);
        if (sym_info != .int or sym_info.int.signedness != .unsigned)
            @compileError("Dafsa: SymbolType must be an unsigned integer type, got " ++ @typeName(SymbolType));

        const idx_info = @typeInfo(IndexType);
        if (idx_info != .int or idx_info.int.signedness != .unsigned)
            @compileError("Dafsa: IndexType must be an unsigned integer type, got " ++ @typeName(IndexType));
        if (@bitSizeOf(IndexType) < 16)
            @compileError("Dafsa: IndexType must be at least u16");
        if (@bitSizeOf(IndexType) > @bitSizeOf(usize))
            @compileError("Dafsa: IndexType (" ++ @typeName(IndexType) ++ ") is wider than usize on this target — use a narrower IndexType");
    }

    const symbol_size = @sizeOf(SymbolType);
    const index_size = @sizeOf(IndexType);
    const edge_size = symbol_size + index_size;

    // DegreeType must hold values 0..2^symbol_bits, so it needs symbol_bits+1 bits,
    // rounded up to the next standard unsigned integer width.
    const symbol_bits = @bitSizeOf(SymbolType);
    const DegreeType_ = std.meta.Int(.unsigned, std.math.ceilPowerOfTwo(usize, symbol_bits + 1) catch symbol_bits + 1);
    const degree_size = @sizeOf(DegreeType_);

    return struct {
        pub const DegreeType = DegreeType_;

        /// Header size for the standard serialized format (num_states + num_edges).
        pub const HEADER_SIZE: usize = 2 * index_size;

        /// Header size for the compact serialized format (magic + num_states + num_edges).
        pub const COMPACT_HEADER_SIZE: usize = 1 + 2 * index_size;

        pub const Error = error{
            EmptyInput,
            UnsortedInput,
            OutOfMemory,
            InvalidData,
            /// Internal algorithm invariant violated during construction.
            InvariantViolated,
        };

        // ── Build-time structures ──

        const BuildEdge = struct {
            label: SymbolType,
            target: IndexType,
        };

        const BuildState = struct {
            edges: std.ArrayListUnmanaged(BuildEdge),
            is_final: bool,

            fn init() BuildState {
                return .{
                    .edges = .{},
                    .is_final = false,
                };
            }

            fn deinit(self: *BuildState, alloc: std.mem.Allocator) void {
                self.edges.deinit(alloc);
            }

            fn findChild(self: *const BuildState, label: SymbolType) ?IndexType {
                for (self.edges.items) |e| {
                    if (e.label == label) return e.target;
                }
                return null;
            }

            fn setChild(self: *BuildState, alloc: std.mem.Allocator, label: SymbolType, target: IndexType) !void {
                for (self.edges.items) |*e| {
                    if (e.label == label) {
                        e.target = target;
                        return;
                    }
                }
                try self.edges.append(alloc, .{ .label = label, .target = target });
            }

            fn lastChild(self: *const BuildState) ?BuildEdge {
                if (self.edges.items.len == 0) return null;
                return self.edges.items[self.edges.items.len - 1];
            }

            /// Signature for hashing: (is_final, sorted_edges).
            fn eql(a: *const BuildState, b: *const BuildState) bool {
                if (a.is_final != b.is_final) return false;
                if (a.edges.items.len != b.edges.items.len) return false;
                for (a.edges.items, b.edges.items) |ea, eb| {
                    if (ea.label != eb.label or ea.target != eb.target) return false;
                }
                return true;
            }

            fn hash(self: *const BuildState) u64 {
                var h: u64 = if (self.is_final) 1 else 0;
                for (self.edges.items) |e| {
                    h = h *% 31 +% @as(u64, e.label);
                    h = h *% 31 +% @as(u64, e.target);
                }
                return h;
            }
        };

        // ── Build ──

        /// Build a DAFSA from a sorted list of symbol sequences.
        /// Returns serialized data on `arena.output`.
        pub fn build(arena: ArenaConfig, strings: []const []const SymbolType) ![]u8 {
            return buildImpl(arena.temp, arena.output, strings);
        }

        /// Build a DAFSA using a single allocator for both scratch and output.
        /// Convenience wrapper around `build(ArenaConfig.uniform(allocator), ...)`.
        pub fn buildUniform(allocator: std.mem.Allocator, strings: []const []const SymbolType) ![]u8 {
            return buildImpl(allocator, allocator, strings);
        }

        fn buildImpl(
            temp: std.mem.Allocator,
            output: std.mem.Allocator,
            strings: []const []const SymbolType,
        ) ![]u8 {
            if (strings.len == 0) return Error.EmptyInput;

            // Verify sorted order
            for (1..strings.len) |i| {
                if (std.mem.order(SymbolType, strings[i], strings[i - 1]) != .gt) {
                    return Error.UnsortedInput;
                }
            }

            // Incremental DAFSA construction (Daciuk et al.)
            var states = std.ArrayListUnmanaged(BuildState){};
            defer {
                for (states.items) |*s| s.deinit(temp);
                states.deinit(temp);
            }

            // Root state = index 0
            try states.append(temp, BuildState.init());

            // Register: maps state signature → canonical state index
            var register = std.AutoHashMapUnmanaged(u64, IndexType){};
            defer register.deinit(temp);

            var prev_word: []const SymbolType = &.{};

            for (strings) |word| {
                // Find common prefix length with previous word
                var prefix_len: usize = 0;
                while (prefix_len < prev_word.len and prefix_len < word.len) {
                    if (prev_word[prefix_len] != word[prefix_len]) break;
                    prefix_len += 1;
                }

                // Minimize states after the common prefix of the previous word
                if (prev_word.len > 0) {
                    try minimizeSuffix(&states, &register, temp, prev_word, prefix_len);
                }

                // Add suffix of current word
                var cur_state: IndexType = 0;
                for (0..prefix_len) |ci| {
                    cur_state = states.items[cur_state].findChild(word[ci]) orelse
                        return Error.InvariantViolated;
                }

                for (prefix_len..word.len) |ci| {
                    if (states.items.len > std.math.maxInt(IndexType))
                        return Error.InvalidData;
                    const new_idx: IndexType = @intCast(states.items.len);
                    try states.append(temp, BuildState.init());
                    try states.items[cur_state].setChild(temp, word[ci], new_idx);
                    cur_state = new_idx;
                }

                states.items[cur_state].is_final = true;
                prev_word = word;
            }

            // Final minimization for the last word
            if (prev_word.len > 0) {
                try minimizeSuffix(&states, &register, temp, prev_word, 0);
            }

            // Build a mapping from old state indices to new compacted indices
            if (states.items.len > std.math.maxInt(IndexType))
                return Error.InvalidData;
            const num_states: IndexType = @intCast(states.items.len);

            // Count total edges
            var total_edges_usize: usize = 0;
            for (states.items) |*s| {
                total_edges_usize += s.edges.items.len;
            }
            if (total_edges_usize > std.math.maxInt(IndexType))
                return Error.InvalidData;
            const total_edges: IndexType = @intCast(total_edges_usize);

            // Serialize
            // Header: num_states(index_size) + num_edges(index_size)
            // + edge_offsets(num_states × index_size) + is_final(num_states × 1)
            // + edges(num_edges × edge_size)
            // Root is always state 0 (no footer needed).
            const header_size = HEADER_SIZE;
            const offsets_size: usize = @as(usize, num_states) * index_size;
            const finals_size: usize = @as(usize, num_states);
            const edges_total_size: usize = @as(usize, total_edges) * edge_size;
            const total_size = header_size + offsets_size + finals_size + edges_total_size;

            const result = try output.alloc(u8, total_size);

            var off: usize = 0;
            std.mem.writeInt(IndexType, result[off..][0..index_size], num_states, .little);
            off += index_size;
            std.mem.writeInt(IndexType, result[off..][0..index_size], total_edges, .little);
            off += index_size;

            // Edge offsets
            const offsets_start = off;
            var edge_off: IndexType = 0;
            for (0..num_states) |i| {
                std.mem.writeInt(IndexType, result[offsets_start + i * index_size ..][0..index_size], edge_off, .little);
                edge_off += @intCast(states.items[i].edges.items.len);
            }
            off += offsets_size;

            // Is-final flags
            for (0..num_states) |i| {
                result[off + i] = if (states.items[i].is_final) 1 else 0;
            }
            off += finals_size;

            // Edges: (SymbolType label, IndexType target) per edge
            for (states.items) |*s| {
                for (s.edges.items) |e| {
                    std.mem.writeInt(SymbolType, result[off..][0..symbol_size], e.label, .little);
                    off += symbol_size;
                    std.mem.writeInt(IndexType, result[off..][0..index_size], e.target, .little);
                    off += index_size;
                }
            }

            return result;
        }

        fn minimizeSuffix(
            states: *std.ArrayListUnmanaged(BuildState),
            register: *std.AutoHashMapUnmanaged(u64, IndexType),
            alloc: std.mem.Allocator,
            word: []const SymbolType,
            start: usize,
        ) !void {
            // Walk from the end of the word back to start, minimizing each state
            var i = word.len;
            while (i > start) {
                i -= 1;

                // Find the parent state and the child state
                var parent: IndexType = 0;
                for (0..i) |ci| {
                    parent = states.items[parent].findChild(word[ci]) orelse
                        return Error.InvariantViolated;
                }

                const child = states.items[parent].findChild(word[i]) orelse
                    return Error.InvariantViolated;
                const child_hash = states.items[child].hash();

                if (register.get(child_hash)) |existing| {
                    if (existing != child and BuildState.eql(&states.items[existing], &states.items[child])) {
                        // Replace child with existing registered state
                        try states.items[parent].setChild(alloc, word[i], existing);
                    }
                } else {
                    try register.put(alloc, child_hash, child);
                }
            }
        }

        // ── Query-time (zero-alloc on []const u8) ──

        /// Check if a symbol sequence is in the DAFSA.
        /// Unchecked: assumes valid data. Use `validate()` first for untrusted data.
        pub fn contains(data: []const u8, word: []const SymbolType) bool {
            var state = getRoot(data);
            for (word) |ch| {
                if (findEdge(data, state, ch)) |target| {
                    state = target;
                } else {
                    return false;
                }
            }
            return isFinal(data, state);
        }

        /// Check if any sequence in the DAFSA has the given prefix.
        /// Unchecked: assumes valid data. Use `validate()` first for untrusted data.
        pub fn hasPrefix(data: []const u8, prefix: []const SymbolType) bool {
            var state = getRoot(data);
            for (prefix) |ch| {
                if (findEdge(data, state, ch)) |target| {
                    state = target;
                } else {
                    return false;
                }
            }
            return true;
        }

        /// Number of states in the DAFSA.
        /// Unchecked: assumes valid data. Use `validate()` first for untrusted data.
        pub fn numStates(data: []const u8) IndexType {
            return std.mem.readInt(IndexType, data[0..index_size], .little);
        }

        /// Number of edges in the DAFSA.
        /// Unchecked: assumes valid data. Use `validate()` first for untrusted data.
        pub fn numEdges(data: []const u8) IndexType {
            return std.mem.readInt(IndexType, data[index_size..][0..index_size], .little);
        }

        /// Validate encoded data from an untrusted source.
        /// After a successful `validate()`, `contains()` and `hasPrefix()` are
        /// guaranteed safe. Checks header bounds, edge offset monotonicity,
        /// edge target validity, and finals flags.
        pub fn validate(data: []const u8) Error!void {
            const header_size = HEADER_SIZE;
            if (data.len < header_size) return Error.InvalidData;

            const ns: usize = numStates(data);
            const ne: usize = numEdges(data);

            if (ns == 0) return Error.InvalidData;

            const offsets_size = std.math.mul(usize, ns, index_size) catch return Error.InvalidData;
            const finals_size: usize = ns;
            const edges_total_size = std.math.mul(usize, ne, edge_size) catch return Error.InvalidData;
            const expected_size = std.math.add(usize, header_size, offsets_size) catch return Error.InvalidData;
            const expected_size2 = std.math.add(usize, expected_size, finals_size) catch return Error.InvalidData;
            const expected_size3 = std.math.add(usize, expected_size2, edges_total_size) catch return Error.InvalidData;

            if (data.len != expected_size3) return Error.InvalidData;

            // Validate edge offsets: monotonically non-decreasing, within bounds.
            const offsets_start: usize = header_size;
            var prev_off: IndexType = 0;
            for (0..ns) |i| {
                const off = std.mem.readInt(IndexType, data[offsets_start + i * index_size ..][0..index_size], .little);
                if (i > 0 and off < prev_off) return Error.InvalidData;
                if (off > ne) return Error.InvalidData;
                prev_off = off;
            }

            // Validate finals flags: must be 0 or 1.
            const finals_start: usize = header_size + offsets_size;
            for (0..ns) |i| {
                if (data[finals_start + i] > 1) return Error.InvalidData;
            }

            // Validate edge targets: must be < num_states.
            const edges_start: usize = header_size + offsets_size + finals_size;
            for (0..ne) |i| {
                const target_off = edges_start + i * edge_size + symbol_size;
                const target = std.mem.readInt(IndexType, data[target_off..][0..index_size], .little);
                if (target >= ns) return Error.InvalidData;
            }
        }

        // ── Internal helpers ──

        fn getRoot(_: []const u8) IndexType {
            return 0; // Root is always state 0 by construction.
        }

        fn isFinal(data: []const u8, state: IndexType) bool {
            const ns: usize = numStates(data);
            const finals_off: usize = 2 * index_size + ns * index_size;
            return data[finals_off + state] != 0;
        }

        fn getEdgeOffset(data: []const u8, state: IndexType) IndexType {
            const off = 2 * index_size + @as(usize, state) * index_size;
            return std.mem.readInt(IndexType, data[off..][0..index_size], .little);
        }

        fn getEdgeCount(data: []const u8, state: IndexType) IndexType {
            const ns = numStates(data);
            const next_off = if (state + 1 < ns) getEdgeOffset(data, state + 1) else numEdges(data);
            return next_off - getEdgeOffset(data, state);
        }

        fn findEdge(data: []const u8, state: IndexType, label: SymbolType) ?IndexType {
            const ns: usize = numStates(data);
            const edge_base: usize = 2 * index_size + ns * index_size + ns; // start of edges section
            const start = getEdgeOffset(data, state);
            const count = getEdgeCount(data, state);

            for (0..count) |i| {
                const e_off = edge_base + (@as(usize, start) + i) * edge_size;
                const e_label = std.mem.readInt(SymbolType, data[e_off..][0..symbol_size], .little);
                if (e_label == label) {
                    return std.mem.readInt(IndexType, data[e_off + symbol_size ..][0..index_size], .little);
                }
            }
            return null;
        }

        // ═══════════════════════════════════════════════════════════
        // Compact serialization — optimized for small automata.
        //
        // Format (all little-endian):
        //   [u8: magic (0xDA)]
        //   [IndexType: num_states]
        //   [IndexType: num_edges]
        //   [num_states × DegreeType: degree per state]
        //   [ceil(num_states/8) bytes: finals bitvector]
        //   [num_edges × (SymbolType + IndexType) bytes: edges]
        //
        // DegreeType is derived from SymbolType — wide enough to hold
        // 0..2^symbol_bits (e.g. u16 for u8 symbols, u32 for u21 symbols).
        //
        // Savings vs standard format:
        //   - Bitvector for finals (vs byte-per-state)
        //   - Degree-per-state (vs IndexType-sized edge offset table)
        //   - No root_state footer (root is always state 0)
        // ═══════════════════════════════════════════════════════════

        const COMPACT_MAGIC: u8 = 0xDA;
        const compact_header_size = COMPACT_HEADER_SIZE;

        /// Build a compact DAFSA from a sorted list of symbol sequences.
        pub fn buildCompact(arena: ArenaConfig, strings: []const []const SymbolType) ![]u8 {
            return buildCompactImpl(arena.temp, arena.output, strings);
        }

        /// Build a compact DAFSA using a single allocator.
        /// Convenience wrapper around `buildCompact(ArenaConfig.uniform(allocator), ...)`.
        pub fn buildCompactUniform(allocator: std.mem.Allocator, strings: []const []const SymbolType) ![]u8 {
            return buildCompactImpl(allocator, allocator, strings);
        }

        fn buildCompactImpl(
            temp: std.mem.Allocator,
            output: std.mem.Allocator,
            strings: []const []const SymbolType,
        ) ![]u8 {
            // First build the standard DAFSA to get the minimized automaton
            const standard = try buildImpl(temp, temp, strings);
            defer temp.free(standard);

            const ns: usize = numStates(standard);
            const ne: usize = numEdges(standard);

            const num_states_idx: IndexType = @intCast(ns);
            const num_edges_idx: IndexType = @intCast(ne);

            // Calculate sizes
            const finals_bytes = (ns + 7) / 8;
            const degrees_size: usize = ns * degree_size;
            const edges_total_size: usize = ne * edge_size;
            const total_size = compact_header_size + degrees_size + finals_bytes + edges_total_size;

            const result = try output.alloc(u8, total_size);

            var off: usize = 0;

            // Header
            result[off] = COMPACT_MAGIC;
            off += 1;
            std.mem.writeInt(IndexType, result[off..][0..index_size], num_states_idx, .little);
            off += index_size;
            std.mem.writeInt(IndexType, result[off..][0..index_size], num_edges_idx, .little);
            off += index_size;

            // Degrees (DegreeType per state)
            for (0..ns) |i| {
                const s: IndexType = @intCast(i);
                const edge_count = getEdgeCount(standard, s);
                if (edge_count > std.math.maxInt(DegreeType)) return Error.InvalidData;
                std.mem.writeInt(DegreeType, result[off..][0..degree_size], @intCast(edge_count), .little);
                off += degree_size;
            }

            // Finals bitvector
            const finals_start = off;
            @memset(result[finals_start..][0..finals_bytes], 0);
            for (0..ns) |i| {
                const s: IndexType = @intCast(i);
                if (isFinal(standard, s)) {
                    const byte_idx = i / 8;
                    const bit_idx: u3 = @intCast(i % 8);
                    result[finals_start + byte_idx] |= @as(u8, 1) << bit_idx;
                }
            }
            off += finals_bytes;

            // Edges: (SymbolType label, IndexType target), in state order
            const old_edge_base: usize = 2 * index_size + ns * index_size + ns;
            for (0..ne) |i| {
                const e_off = old_edge_base + i * edge_size;
                const lbl = std.mem.readInt(SymbolType, standard[e_off..][0..symbol_size], .little);
                std.mem.writeInt(SymbolType, result[off..][0..symbol_size], lbl, .little);
                off += symbol_size;
                const target = std.mem.readInt(IndexType, standard[e_off + symbol_size ..][0..index_size], .little);
                std.mem.writeInt(IndexType, result[off..][0..index_size], target, .little);
                off += index_size;
            }

            return result;
        }

        // ── Compact query-time (zero-alloc on []const u8) ──

        /// Check if a symbol sequence is in the compact DAFSA.
        /// Unchecked: assumes valid data. Use `validateCompact()` first for untrusted data.
        pub fn containsCompact(data: []const u8, word: []const SymbolType) bool {
            var state: IndexType = 0; // root is always state 0
            for (word) |ch| {
                if (findEdgeCompact(data, state, ch)) |target| {
                    state = target;
                } else {
                    return false;
                }
            }
            return isFinalCompact(data, state);
        }

        /// Check if any sequence in the compact DAFSA has the given prefix.
        /// Unchecked: assumes valid data. Use `validateCompact()` first for untrusted data.
        pub fn hasPrefixCompact(data: []const u8, prefix: []const SymbolType) bool {
            var state: IndexType = 0;
            for (prefix) |ch| {
                if (findEdgeCompact(data, state, ch)) |target| {
                    state = target;
                } else {
                    return false;
                }
            }
            return true;
        }

        /// Number of states in compact format.
        pub fn numStatesCompact(data: []const u8) IndexType {
            return std.mem.readInt(IndexType, data[1..][0..index_size], .little);
        }

        /// Number of edges in compact format.
        pub fn numEdgesCompact(data: []const u8) IndexType {
            return std.mem.readInt(IndexType, data[1 + index_size ..][0..index_size], .little);
        }

        /// Validate compact encoded data from an untrusted source.
        pub fn validateCompact(data: []const u8) Error!void {
            if (data.len < compact_header_size) return Error.InvalidData;
            if (data[0] != COMPACT_MAGIC) return Error.InvalidData;

            const ns: usize = numStatesCompact(data);
            const ne: usize = numEdgesCompact(data);

            if (ns == 0) return Error.InvalidData;

            const finals_bytes = (ns + 7) / 8;
            const degrees_size = std.math.mul(usize, ns, degree_size) catch return Error.InvalidData;
            const edges_total_size = std.math.mul(usize, ne, edge_size) catch return Error.InvalidData;
            const expected_size = std.math.add(usize, compact_header_size, degrees_size) catch return Error.InvalidData;
            const expected_size2 = std.math.add(usize, expected_size, finals_bytes) catch return Error.InvalidData;
            const expected_size3 = std.math.add(usize, expected_size2, edges_total_size) catch return Error.InvalidData;

            if (data.len != expected_size3) return Error.InvalidData;

            // Validate degrees sum to num_edges.
            var degree_sum: usize = 0;
            for (0..ns) |i| {
                degree_sum += std.mem.readInt(DegreeType, data[compact_header_size + i * degree_size ..][0..degree_size], .little);
            }
            if (degree_sum != ne) return Error.InvalidData;

            // Validate edge targets.
            const edges_start = compact_header_size + degrees_size + finals_bytes;
            for (0..ne) |i| {
                const target_off = edges_start + i * edge_size + symbol_size;
                const target = std.mem.readInt(IndexType, data[target_off..][0..index_size], .little);
                if (target >= ns) return Error.InvalidData;
            }
        }

        fn isFinalCompact(data: []const u8, state: IndexType) bool {
            const ns: usize = numStatesCompact(data);
            const finals_off: usize = compact_header_size + ns * degree_size; // after header + degrees
            const byte_idx = @as(usize, state) / 8;
            const bit_idx: u3 = @intCast(@as(usize, state) % 8);
            return (data[finals_off + byte_idx] & (@as(u8, 1) << bit_idx)) != 0;
        }

        fn findEdgeCompact(data: []const u8, state: IndexType, label: SymbolType) ?IndexType {
            const ns: usize = numStatesCompact(data);
            const finals_bytes = (ns + 7) / 8;
            const edges_start: usize = compact_header_size + ns * degree_size + finals_bytes;

            // Compute cumulative edge offset for this state by summing degrees
            var edge_off: usize = 0;
            for (0..state) |i| {
                edge_off += std.mem.readInt(DegreeType, data[compact_header_size + i * degree_size ..][0..degree_size], .little);
            }
            const degree_val = std.mem.readInt(DegreeType, data[compact_header_size + @as(usize, state) * degree_size ..][0..degree_size], .little);

            // Scan edges for this state
            for (0..degree_val) |i| {
                const e_off = edges_start + (edge_off + i) * edge_size;
                const e_label = std.mem.readInt(SymbolType, data[e_off..][0..symbol_size], .little);
                if (e_label == label) {
                    return std.mem.readInt(IndexType, data[e_off + symbol_size ..][0..index_size], .little);
                }
            }
            return null;
        }
    };
}

/// Re-export from `comptime_utils.zig` for convenience.
/// See `comptime_utils.detectSymbolType` for full documentation.
pub const detectSymbolType = @import("../comptime_utils.zig").detectSymbolType;

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;
const ByteDafsa = Dafsa(u8, u32);

test "DAFSA: basic membership" {
    const strings = [_][]const u8{ "cat", "cats", "dog", "dogs" };
    const data = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.contains(data, "cat"));
    try testing.expect(ByteDafsa.contains(data, "cats"));
    try testing.expect(ByteDafsa.contains(data, "dog"));
    try testing.expect(ByteDafsa.contains(data, "dogs"));

    try testing.expect(!ByteDafsa.contains(data, "ca"));
    try testing.expect(!ByteDafsa.contains(data, "do"));
    try testing.expect(!ByteDafsa.contains(data, "bat"));
    try testing.expect(!ByteDafsa.contains(data, ""));
}

test "DAFSA: prefix query" {
    const strings = [_][]const u8{ "apple", "application", "apply" };
    const data = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.hasPrefix(data, "app"));
    try testing.expect(ByteDafsa.hasPrefix(data, "appl"));
    try testing.expect(ByteDafsa.hasPrefix(data, "apple"));
    try testing.expect(!ByteDafsa.hasPrefix(data, "b"));
    try testing.expect(!ByteDafsa.hasPrefix(data, "az"));
}

test "DAFSA: single string" {
    const strings = [_][]const u8{"hello"};
    const data = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.contains(data, "hello"));
    try testing.expect(!ByteDafsa.contains(data, "hell"));
    try testing.expect(!ByteDafsa.contains(data, "helloo"));
}

test "DAFSA: suffix sharing" {
    // "listing" and "testing" share suffix "ting"
    const strings = [_][]const u8{ "listing", "testing" };
    const data = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.contains(data, "listing"));
    try testing.expect(ByteDafsa.contains(data, "testing"));
    try testing.expect(!ByteDafsa.contains(data, "ting"));
    try testing.expect(!ByteDafsa.contains(data, "list"));
    try testing.expect(!ByteDafsa.contains(data, "test"));
}

test "DAFSA: many words with shared suffixes" {
    const strings = [_][]const u8{
        "acted",
        "baked",
        "biked",
        "hiked",
        "liked",
    };
    const data = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    for (strings) |w| {
        try testing.expect(ByteDafsa.contains(data, w));
    }
    try testing.expect(!ByteDafsa.contains(data, "ked"));
    try testing.expect(!ByteDafsa.contains(data, "bake"));
}

test "DAFSA: empty input returns error" {
    const strings = [_][]const u8{};
    try testing.expectError(error.EmptyInput, ByteDafsa.buildUniform(testing.allocator, &strings));
}

test "DAFSA: unsorted input returns error" {
    const strings = [_][]const u8{ "banana", "apple" };
    try testing.expectError(error.UnsortedInput, ByteDafsa.buildUniform(testing.allocator, &strings));
}

test "DAFSA: state and edge counts" {
    const strings = [_][]const u8{ "a", "b", "c" };
    const data = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    // Root + 3 leaf states (but "a", "b", "c" may share a final state)
    try testing.expect(ByteDafsa.numStates(data) >= 2);
    try testing.expect(ByteDafsa.numEdges(data) == 3);

    for (strings) |w| {
        try testing.expect(ByteDafsa.contains(data, w));
    }
}

test "DAFSA: ArenaConfig variant" {
    const arena = ArenaConfig.uniform(testing.allocator);
    const strings = [_][]const u8{ "foo", "foobar", "foobaz" };
    const data = try ByteDafsa.build(arena, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.contains(data, "foo"));
    try testing.expect(ByteDafsa.contains(data, "foobar"));
    try testing.expect(ByteDafsa.contains(data, "foobaz"));
    try testing.expect(!ByteDafsa.contains(data, "fooba"));
    try testing.expect(!ByteDafsa.contains(data, "foobarr"));
}

test "DAFSA: validate accepts valid data" {
    const strings = [_][]const u8{ "cat", "cats", "dog", "dogs" };
    const data = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try ByteDafsa.validate(data);
}

test "DAFSA: validate rejects too-short blob" {
    const short = [_]u8{ 0, 0, 0 };
    try testing.expectError(ByteDafsa.Error.InvalidData, ByteDafsa.validate(&short));
}

test "DAFSA: validate rejects truncated blob" {
    const strings = [_][]const u8{ "cat", "dog" };
    const data = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    // Truncate by removing last few bytes
    try testing.expectError(ByteDafsa.Error.InvalidData, ByteDafsa.validate(data[0 .. data.len - 2]));
}

// ── Compact format tests ──

test "DAFSA compact: basic membership" {
    const strings = [_][]const u8{ "cat", "cats", "dog", "dogs" };
    const data = try ByteDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.containsCompact(data, "cat"));
    try testing.expect(ByteDafsa.containsCompact(data, "cats"));
    try testing.expect(ByteDafsa.containsCompact(data, "dog"));
    try testing.expect(ByteDafsa.containsCompact(data, "dogs"));

    try testing.expect(!ByteDafsa.containsCompact(data, "ca"));
    try testing.expect(!ByteDafsa.containsCompact(data, "do"));
    try testing.expect(!ByteDafsa.containsCompact(data, "bat"));
    try testing.expect(!ByteDafsa.containsCompact(data, ""));
}

test "DAFSA compact: prefix query" {
    const strings = [_][]const u8{ "apple", "application", "apply" };
    const data = try ByteDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.hasPrefixCompact(data, "app"));
    try testing.expect(ByteDafsa.hasPrefixCompact(data, "appl"));
    try testing.expect(ByteDafsa.hasPrefixCompact(data, "apple"));
    try testing.expect(!ByteDafsa.hasPrefixCompact(data, "b"));
    try testing.expect(!ByteDafsa.hasPrefixCompact(data, "az"));
}

test "DAFSA compact: suffix sharing" {
    const strings = [_][]const u8{ "listing", "testing" };
    const data = try ByteDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.containsCompact(data, "listing"));
    try testing.expect(ByteDafsa.containsCompact(data, "testing"));
    try testing.expect(!ByteDafsa.containsCompact(data, "ting"));
    try testing.expect(!ByteDafsa.containsCompact(data, "list"));
}

test "DAFSA compact: many words with shared suffixes" {
    const strings = [_][]const u8{
        "acted",
        "baked",
        "biked",
        "hiked",
        "liked",
    };
    const data = try ByteDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    for (strings) |w| {
        try testing.expect(ByteDafsa.containsCompact(data, w));
    }
    try testing.expect(!ByteDafsa.containsCompact(data, "ked"));
    try testing.expect(!ByteDafsa.containsCompact(data, "bake"));
}

test "DAFSA compact: single string" {
    const strings = [_][]const u8{"hello"};
    const data = try ByteDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(ByteDafsa.containsCompact(data, "hello"));
    try testing.expect(!ByteDafsa.containsCompact(data, "hell"));
    try testing.expect(!ByteDafsa.containsCompact(data, "helloo"));
}

test "DAFSA compact: size vs standard" {
    const strings = [_][]const u8{
        "acted",
        "baked",
        "biked",
        "hiked",
        "liked",
    };
    const standard = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(standard);
    const compact = try ByteDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(compact);

    // Compact must be smaller
    try testing.expect(compact.len < standard.len);
}

test "DAFSA compact: validate accepts valid data" {
    const strings = [_][]const u8{ "cat", "cats", "dog", "dogs" };
    const data = try ByteDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try ByteDafsa.validateCompact(data);
}

test "DAFSA compact: validate rejects bad magic" {
    const strings = [_][]const u8{ "cat", "dog" };
    const data = try ByteDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    var corrupt = try testing.allocator.alloc(u8, data.len);
    defer testing.allocator.free(corrupt);
    @memcpy(corrupt, data);
    corrupt[0] = 0xFF;
    try testing.expectError(ByteDafsa.Error.InvalidData, ByteDafsa.validateCompact(corrupt));
}

// ── u16 IndexType tests ──

test "DAFSA: u16 IndexType round-trip" {
    const SmallDafsa = Dafsa(u8, u16);
    const strings = [_][]const u8{ "cat", "cats", "dog", "dogs" };
    const data = try SmallDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(SmallDafsa.contains(data, "cat"));
    try testing.expect(SmallDafsa.contains(data, "cats"));
    try testing.expect(SmallDafsa.contains(data, "dog"));
    try testing.expect(SmallDafsa.contains(data, "dogs"));
    try testing.expect(!SmallDafsa.contains(data, "bat"));
    try SmallDafsa.validate(data);
}

test "DAFSA: u16 IndexType compact" {
    const SmallDafsa = Dafsa(u8, u16);
    const strings = [_][]const u8{ "apple", "application", "apply" };
    const data = try SmallDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(SmallDafsa.containsCompact(data, "apple"));
    try testing.expect(SmallDafsa.containsCompact(data, "application"));
    try testing.expect(SmallDafsa.containsCompact(data, "apply"));
    try testing.expect(!SmallDafsa.containsCompact(data, "app"));
    try SmallDafsa.validateCompact(data);
}

test "DAFSA: u16 IndexType is smaller than u32" {
    const SmallDafsa = Dafsa(u8, u16);
    const strings = [_][]const u8{
        "acted",
        "baked",
        "biked",
        "hiked",
        "liked",
    };
    const small = try SmallDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(small);
    const large = try ByteDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(large);

    // u16 version must be smaller
    try testing.expect(small.len < large.len);
}

// ── u64 IndexType tests ──

test "DAFSA: u64 IndexType round-trip" {
    const LargeDafsa = Dafsa(u8, u64);
    const strings = [_][]const u8{ "cat", "cats", "dog", "dogs" };
    const data = try LargeDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(LargeDafsa.contains(data, "cat"));
    try testing.expect(LargeDafsa.contains(data, "cats"));
    try testing.expect(LargeDafsa.contains(data, "dog"));
    try testing.expect(LargeDafsa.contains(data, "dogs"));
    try testing.expect(!LargeDafsa.contains(data, "bat"));
    try LargeDafsa.validate(data);
}

test "DAFSA: u64 IndexType compact" {
    const LargeDafsa = Dafsa(u8, u64);
    const strings = [_][]const u8{ "apple", "application", "apply" };
    const data = try LargeDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(LargeDafsa.containsCompact(data, "apple"));
    try testing.expect(LargeDafsa.containsCompact(data, "application"));
    try testing.expect(LargeDafsa.containsCompact(data, "apply"));
    try testing.expect(!LargeDafsa.containsCompact(data, "app"));
    try LargeDafsa.validateCompact(data);
}

// ── u16 SymbolType tests ──

test "DAFSA: u16 SymbolType for token IDs" {
    const TokenDafsa = Dafsa(u16, u32);
    const seq1 = [_]u16{ 100, 200, 300 };
    const seq2 = [_]u16{ 100, 200, 400 };
    const seq3 = [_]u16{ 100, 500 };
    const strings = [_][]const u16{ &seq1, &seq2, &seq3 };
    const data = try TokenDafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(TokenDafsa.contains(data, &seq1));
    try testing.expect(TokenDafsa.contains(data, &seq2));
    try testing.expect(TokenDafsa.contains(data, &seq3));
    try testing.expect(!TokenDafsa.contains(data, &[_]u16{ 100, 200 }));
    try testing.expect(!TokenDafsa.contains(data, &[_]u16{999}));
    try TokenDafsa.validate(data);
}

test "DAFSA: u16 SymbolType compact round-trip" {
    const TokenDafsa = Dafsa(u16, u32);
    const seq1 = [_]u16{ 100, 200, 300 };
    const seq2 = [_]u16{ 100, 200, 400 };
    const seq3 = [_]u16{ 100, 500 };
    const strings = [_][]const u16{ &seq1, &seq2, &seq3 };
    const data = try TokenDafsa.buildCompactUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(TokenDafsa.containsCompact(data, &seq1));
    try testing.expect(TokenDafsa.containsCompact(data, &seq2));
    try testing.expect(TokenDafsa.containsCompact(data, &seq3));
    try testing.expect(!TokenDafsa.containsCompact(data, &[_]u16{ 100, 200 }));
    try TokenDafsa.validateCompact(data);
}

// ── DegreeType derivation tests ──

test "DAFSA: DegreeType is derived correctly" {
    // u8 symbols → max degree 256 → needs u16
    try testing.expect(Dafsa(u8, u32).DegreeType == u16);
    // u5 symbols → max degree 32 → needs u8 (ceil power of 2 of 6 bits = 8)
    try testing.expect(Dafsa(u5, u32).DegreeType == u8);
    // u16 symbols → max degree 65536 → needs u32
    try testing.expect(Dafsa(u16, u32).DegreeType == u32);
    // u21 symbols → max degree 2M → needs u32 (ceil power of 2 of 22 bits = 32)
    try testing.expect(Dafsa(u21, u32).DegreeType == u32);
    // u1 symbols → max degree 2 → needs u2
    try testing.expect(Dafsa(u1, u16).DegreeType == u2);
}

// detectSymbolType tests live in src/comptime_utils.zig
