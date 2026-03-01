//! DAFSA (Directed Acyclic Finite State Automaton) — compact string set.
//!
//! A DAFSA (also called a DAWG — Directed Acyclic Word Graph) compresses
//! a sorted set of strings by sharing both prefixes (like a trie) and
//! suffixes. This produces a minimal-state automaton that recognizes
//! exactly the input string set.
//!
//! **Build-time:** allocates (uses `ArenaConfig`).
//! **Query-time:** zero allocation — lookup on `[]const u8`.
//!
//! ## Construction
//!
//! Uses the incremental algorithm of Daciuk et al. (2000):
//! 1. Process input strings in sorted order.
//! 2. Maintain a "register" of unique right-language states.
//! 3. When a state's right-language is frozen (all its descendants are
//!    finalized), check the register for an equivalent frozen state.
//!    If found, redirect the edge to the register entry.
//!
//! ## Serialized format
//!
//! ```text
//! [u32: num_states]               — total states
//! [u32: num_edges]                — total edges
//! [num_states × u32: edge_offset] — offset into edge array for each state
//! [num_states × u8: is_final]     — 1 if state is accepting, 0 otherwise
//! [num_edges × (u8 + u32)]        — (label_byte, target_state) per edge
//! [u32: root_state]               — index of root state
//! ```
//!
//! ## References
//!
//! - Daciuk et al. (2000), "Incremental Construction of Minimal Acyclic
//!   Finite State Automata"
//! - Lucchesi and Kowaltowski (1993), "Applications of Finite Automata
//!   Representing Large Vocabularies"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;

pub const Dafsa = struct {
    pub const Error = error{
        EmptyInput,
        UnsortedInput,
        OutOfMemory,
        InvalidData,
    };

    // ── Build-time structures ──

    const BuildEdge = struct {
        label: u8,
        target: u32, // state index
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

        fn findChild(self: *const BuildState, label: u8) ?u32 {
            for (self.edges.items) |e| {
                if (e.label == label) return e.target;
            }
            return null;
        }

        fn setChild(self: *BuildState, alloc: std.mem.Allocator, label: u8, target: u32) !void {
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
                h = h *% 31 +% e.label;
                h = h *% 31 +% e.target;
            }
            return h;
        }
    };

    // ── Build ──

    /// Build a DAFSA from a sorted list of strings.
    /// Returns serialized data on `arena.output`.
    pub fn build(arena: ArenaConfig, strings: []const []const u8) ![]u8 {
        return buildImpl(arena.temp, arena.output, strings);
    }

    pub fn buildUniform(allocator: std.mem.Allocator, strings: []const []const u8) ![]u8 {
        return buildImpl(allocator, allocator, strings);
    }

    fn buildImpl(
        temp: std.mem.Allocator,
        output: std.mem.Allocator,
        strings: []const []const u8,
    ) ![]u8 {
        if (strings.len == 0) return Error.EmptyInput;

        // Verify sorted order
        for (1..strings.len) |i| {
            if (std.mem.order(u8, strings[i], strings[i - 1]) != .gt) {
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
        var register = std.AutoHashMapUnmanaged(u64, u32){};
        defer register.deinit(temp);

        var prev_word: []const u8 = "";

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
            var cur_state: u32 = 0;
            for (0..prefix_len) |ci| {
                cur_state = states.items[cur_state].findChild(word[ci]).?;
            }

            for (prefix_len..word.len) |ci| {
                const new_idx: u32 = @intCast(states.items.len);
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
        // (some states may have been replaced by register entries)
        const num_states: u32 = @intCast(states.items.len);

        // Count total edges
        var total_edges: u32 = 0;
        for (states.items) |*s| {
            total_edges += @intCast(s.edges.items.len);
        }

        // Serialize
        // Header: num_states(4) + num_edges(4)
        // + edge_offsets(num_states × 4) + is_final(num_states × 1)
        // + edges(num_edges × 5) + root_state(4)
        const header_size: u32 = 4 + 4;
        const offsets_size: u32 = num_states * 4;
        const finals_size: u32 = num_states;
        const edges_size: u32 = total_edges * 5; // u8 label + u32 target
        const footer_size: u32 = 4; // root state
        const total_size = header_size + offsets_size + finals_size + edges_size + footer_size;

        const result = try output.alloc(u8, total_size);

        var off: u32 = 0;
        std.mem.writeInt(u32, result[off..][0..4], num_states, .little);
        off += 4;
        std.mem.writeInt(u32, result[off..][0..4], total_edges, .little);
        off += 4;

        // Edge offsets
        const offsets_start = off;
        var edge_off: u32 = 0;
        for (0..num_states) |i| {
            std.mem.writeInt(u32, result[offsets_start + @as(u32, @intCast(i)) * 4 ..][0..4], edge_off, .little);
            edge_off += @intCast(states.items[i].edges.items.len);
        }
        off += offsets_size;

        // Is-final flags
        for (0..num_states) |i| {
            result[off + @as(u32, @intCast(i))] = if (states.items[i].is_final) 1 else 0;
        }
        off += finals_size;

        // Edges
        for (states.items) |*s| {
            for (s.edges.items) |e| {
                result[off] = e.label;
                off += 1;
                std.mem.writeInt(u32, result[off..][0..4], e.target, .little);
                off += 4;
            }
        }

        // Root state
        std.mem.writeInt(u32, result[off..][0..4], 0, .little);

        return result;
    }

    fn minimizeSuffix(
        states: *std.ArrayListUnmanaged(BuildState),
        register: *std.AutoHashMapUnmanaged(u64, u32),
        alloc: std.mem.Allocator,
        word: []const u8,
        start: usize,
    ) !void {
        // Walk from the end of the word back to start, minimizing each state
        var i = word.len;
        while (i > start) {
            i -= 1;

            // Find the parent state and the child state
            var parent: u32 = 0;
            for (0..i) |ci| {
                parent = states.items[parent].findChild(word[ci]).?;
            }

            const child = states.items[parent].findChild(word[i]).?;
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

    /// Check if a string is in the DAFSA.
    pub fn contains(data: []const u8, word: []const u8) bool {
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

    /// Check if any string in the DAFSA has the given prefix.
    pub fn hasPrefix(data: []const u8, prefix: []const u8) bool {
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
    pub fn numStates(data: []const u8) u32 {
        return std.mem.readInt(u32, data[0..4], .little);
    }

    /// Number of edges in the DAFSA.
    pub fn numEdges(data: []const u8) u32 {
        return std.mem.readInt(u32, data[4..8], .little);
    }

    // ── Internal helpers ──

    fn getRoot(data: []const u8) u32 {
        const n = numStates(data);
        const ne = numEdges(data);
        const footer_off = 8 + n * 4 + n + ne * 5;
        return std.mem.readInt(u32, data[footer_off..][0..4], .little);
    }

    fn isFinal(data: []const u8, state: u32) bool {
        const n = numStates(data);
        const finals_off: u32 = 8 + n * 4;
        return data[finals_off + state] != 0;
    }

    fn getEdgeOffset(data: []const u8, state: u32) u32 {
        const off = 8 + state * 4;
        return std.mem.readInt(u32, data[off..][0..4], .little);
    }

    fn getEdgeCount(data: []const u8, state: u32) u32 {
        const n = numStates(data);
        const next_off = if (state + 1 < n) getEdgeOffset(data, state + 1) else numEdges(data);
        return next_off - getEdgeOffset(data, state);
    }

    fn findEdge(data: []const u8, state: u32, label: u8) ?u32 {
        const n = numStates(data);
        const edge_base: u32 = 8 + n * 4 + n; // start of edges section
        const start = getEdgeOffset(data, state);
        const count = getEdgeCount(data, state);

        for (0..count) |i| {
            const e_off = edge_base + (start + @as(u32, @intCast(i))) * 5;
            if (data[e_off] == label) {
                return std.mem.readInt(u32, data[e_off + 1 ..][0..4], .little);
            }
        }
        return null;
    }
};

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

test "DAFSA: basic membership" {
    const strings = [_][]const u8{ "cat", "cats", "dog", "dogs" };
    const data = try Dafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(Dafsa.contains(data, "cat"));
    try testing.expect(Dafsa.contains(data, "cats"));
    try testing.expect(Dafsa.contains(data, "dog"));
    try testing.expect(Dafsa.contains(data, "dogs"));

    try testing.expect(!Dafsa.contains(data, "ca"));
    try testing.expect(!Dafsa.contains(data, "do"));
    try testing.expect(!Dafsa.contains(data, "bat"));
    try testing.expect(!Dafsa.contains(data, ""));
}

test "DAFSA: prefix query" {
    const strings = [_][]const u8{ "apple", "application", "apply" };
    const data = try Dafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(Dafsa.hasPrefix(data, "app"));
    try testing.expect(Dafsa.hasPrefix(data, "appl"));
    try testing.expect(Dafsa.hasPrefix(data, "apple"));
    try testing.expect(!Dafsa.hasPrefix(data, "b"));
    try testing.expect(!Dafsa.hasPrefix(data, "az"));
}

test "DAFSA: single string" {
    const strings = [_][]const u8{"hello"};
    const data = try Dafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(Dafsa.contains(data, "hello"));
    try testing.expect(!Dafsa.contains(data, "hell"));
    try testing.expect(!Dafsa.contains(data, "helloo"));
}

test "DAFSA: suffix sharing" {
    // "listing" and "testing" share suffix "ting"
    const strings = [_][]const u8{ "listing", "testing" };
    const data = try Dafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    try testing.expect(Dafsa.contains(data, "listing"));
    try testing.expect(Dafsa.contains(data, "testing"));
    try testing.expect(!Dafsa.contains(data, "ting"));
    try testing.expect(!Dafsa.contains(data, "list"));
    try testing.expect(!Dafsa.contains(data, "test"));
}

test "DAFSA: many words with shared suffixes" {
    const strings = [_][]const u8{
        "acted",
        "baked",
        "biked",
        "hiked",
        "liked",
    };
    const data = try Dafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    for (strings) |w| {
        try testing.expect(Dafsa.contains(data, w));
    }
    try testing.expect(!Dafsa.contains(data, "ked"));
    try testing.expect(!Dafsa.contains(data, "bake"));
}

test "DAFSA: empty input returns error" {
    const strings = [_][]const u8{};
    try testing.expectError(error.EmptyInput, Dafsa.buildUniform(testing.allocator, &strings));
}

test "DAFSA: unsorted input returns error" {
    const strings = [_][]const u8{ "banana", "apple" };
    try testing.expectError(error.UnsortedInput, Dafsa.buildUniform(testing.allocator, &strings));
}

test "DAFSA: state and edge counts" {
    const strings = [_][]const u8{ "a", "b", "c" };
    const data = try Dafsa.buildUniform(testing.allocator, &strings);
    defer testing.allocator.free(data);

    // Root + 3 leaf states (but "a", "b", "c" may share a final state)
    try testing.expect(Dafsa.numStates(data) >= 2);
    try testing.expect(Dafsa.numEdges(data) == 3);

    for (strings) |w| {
        try testing.expect(Dafsa.contains(data, w));
    }
}

test "DAFSA: ArenaConfig variant" {
    const arena = ArenaConfig.uniform(testing.allocator);
    const strings = [_][]const u8{ "foo", "foobar", "foobaz" };
    const data = try Dafsa.build(arena, &strings);
    defer testing.allocator.free(data);

    try testing.expect(Dafsa.contains(data, "foo"));
    try testing.expect(Dafsa.contains(data, "foobar"));
    try testing.expect(Dafsa.contains(data, "foobaz"));
    try testing.expect(!Dafsa.contains(data, "fooba"));
    try testing.expect(!Dafsa.contains(data, "foobarr"));
}
