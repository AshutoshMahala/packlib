//! LOUDS (Level-Order Unary Degree Sequence) — succinct tree representation.
//!
//! Encodes an ordered tree with `n` nodes using `2n + 1` bits, supporting O(1)
//! parent, first-child, last-child, next-sibling, and previous-sibling navigation
//! via the underlying rank/select bitvector.
//!
//! **Build-time:** allocates (uses `ArenaConfig`).
//! **Read-time:** zero allocation — all queries on `[]const u8`.
//!
//! ## Encoding
//!
//! Nodes are visited in breadth-first (level) order.  For each node with
//! degree `d`, write `d` one-bits followed by a single zero-bit.  The super-root
//! contributes an initial `10` (one child: the real root).
//!
//! Example: tree with degrees [2, 3, 0, 1, 0] (5 real nodes):
//!
//! ```text
//! super-root(1): 10
//! node A(2):     110
//! node B(3):     1110
//! node C(0):     0
//! node D(1):     10
//! node E(0):     0
//! Full bitstring: 10 110 1110 0 10 0
//! ```
//!
//! ## Serialized format
//!
//! ```text
//! [IndexType: num_nodes]         — number of real nodes (excludes super-root)
//! [rank/select data]             — RankSelect bitvector of length 2n+1
//! [optional: labels]             — node labels (if provided)
//! ```
//!
//! ## Navigation identities
//!
//! In LOUDS, nodes map to 1-bits.  Node `i` (0-indexed) corresponds to the
//! `i`-th 1-bit.  Let `p = select1(i)` be the position of that 1-bit.
//!
//! - `first_child(i)`: `rank1(select0(rank1(p+1) - 1) + 1)`
//! - `next_sibling(i)`: `rank1(p + 1)` if `getBit(p+1) == 1`, else none
//! - `parent(i)`: `select1(rank0(select1(i-1) + 1) - 1)` → then `rank1` of that
//! - `node_degree(i)`: count consecutive 1-bits starting at `select0(i) + 1`
//!
//! ## References
//!
//! - Jacobson (1989), "Space-efficient static trees and graphs"
//! - Benoit et al. (2005), "Representing trees of higher degree"
//! - Arroyuelo et al. (2010), "Succinct trees in practice"

const std = @import("std");
const ArenaConfig = @import("../memory/arena_config.zig").ArenaConfig;
const RankSelect = @import("rank_select.zig").RankSelect;

pub fn Louds(comptime IndexType: type) type {
    comptime {
        const info = @typeInfo(IndexType);
        if (info != .int or info.int.signedness != .unsigned)
            @compileError("Louds: IndexType must be an unsigned integer type, got " ++ @typeName(IndexType));
        if (@sizeOf(IndexType) < 2)
            @compileError("Louds: IndexType must be at least u16");
    }
    const index_size = @sizeOf(IndexType);
    const RS = RankSelect(IndexType);

    return struct {
        const Self = @This();

        pub const Error = error{
            EmptyTree,
            InvalidTree,
            OutOfMemory,
        };

        /// Sentinel value indicating "no such node".
        pub const NONE: IndexType = std.math.maxInt(IndexType);

        // Header: [IndexType: num_nodes]
        const HEADER_SIZE: u32 = index_size;

        // ── Build-time ──

        /// Build a LOUDS bitvector from a breadth-first list of node degrees.
        ///
        /// `degrees` must be given in BFS order: `degrees[0]` is the root's
        /// degree, `degrees[1]` is the first child of root's degree, etc.
        ///
        /// Returns serialized data allocated from `arena.output`.
        pub fn buildFromDegrees(arena: ArenaConfig, degrees: []const IndexType) ![]u8 {
            if (degrees.len == 0) return Error.EmptyTree;

            const n: IndexType = @intCast(degrees.len);

            // Validate tree property FIRST: sum(degrees) must equal n - 1
            var total_children: u64 = 0;
            for (degrees) |d| total_children += d;
            if (total_children != @as(u64, n) - 1) return Error.InvalidTree;

            // Bitvector length = 2n + 1 (super-root "10" + each node: degree ones + one zero)
            const bv_len: u32 = @intCast(@as(u64, n) * 2 + 1);

            // Allocate bit array in temp arena
            const bits = try arena.temp.alloc(u1, bv_len);
            defer arena.temp.free(bits);
            @memset(bits, 0);

            // Super-root: "10" (one child = the real root)
            var pos: u32 = 0;
            bits[pos] = 1;
            pos += 1;
            bits[pos] = 0;
            pos += 1;

            // Each real node: d 1-bits followed by one 0-bit
            for (0..@intCast(n)) |i| {
                const d: u32 = @intCast(degrees[i]);
                for (0..d) |_| {
                    bits[pos] = 1;
                    pos += 1;
                }
                bits[pos] = 0;
                pos += 1;
            }

            // Build rank/select over the bitvector
            const rs_data = try RS.build(arena.temp, bits);
            defer arena.temp.free(rs_data);
            const rs_len: u32 = @intCast(rs_data.len);

            // Assemble output: header + rank/select data
            const total_size = HEADER_SIZE + rs_len;
            const output = try arena.output.alloc(u8, total_size);

            writeIdx(output, 0, n);
            @memcpy(output[HEADER_SIZE..][0..rs_len], rs_data);

            return output;
        }

        /// Convenience: build with a plain allocator.
        pub fn buildFromDegreesUniform(allocator: std.mem.Allocator, degrees: []const IndexType) ![]u8 {
            return buildFromDegrees(ArenaConfig.uniform(allocator), degrees);
        }

        // ── Read-time queries (all no-alloc on []const u8) ──

        /// Number of real nodes in the tree.
        pub fn numNodes(data: []const u8) IndexType {
            return readIdx(data, 0);
        }

        /// Access the rank/select data portion.
        inline fn rsData(data: []const u8) []const u8 {
            return data[HEADER_SIZE..];
        }

        /// Degree (number of children) of node `i`.
        ///
        /// Node i is entity i+1 (entity 0 = super-root).
        /// Entity i+1's terminator = select0(i+1) (0-indexed).
        /// Previous entity's terminator = select0(i).
        /// degree = select0(i+1) - select0(i) - 1
        pub fn degree(data: []const u8, i: IndexType) IndexType {
            const rs = rsData(data);
            const end = RS.select0(rs, i + 1); // node i's 0-terminator
            const prev_end = RS.select0(rs, i); // previous entity's 0-terminator
            return end - prev_end - 1;
        }

        /// First child of node `i`. Returns NONE if node is a leaf.
        ///
        /// Node i's degree-block starts at select0(i) + 1 (0-indexed select).
        /// The 1-bit there maps to node rank1(pos + 1) - 1.
        pub fn firstChild(data: []const u8, i: IndexType) IndexType {
            const rs = rsData(data);
            const block_start = RS.select0(rs, i) + 1;
            const bv_len = RS.totalBits(rs);

            if (block_start >= bv_len) return NONE;
            if (RS.getBit(rs, block_start) == 0) return NONE;

            return RS.rank1(rs, block_start + 1) - 1;
        }

        /// Last child of node `i`. Returns NONE if node is a leaf.
        pub fn lastChild(data: []const u8, i: IndexType) IndexType {
            const rs = rsData(data);
            const block_end = RS.select0(rs, i + 1); // node i's 0-terminator
            const prev_end = RS.select0(rs, i);
            if (block_end <= prev_end + 1) return NONE; // degree 0

            // Last 1-bit is at block_end - 1
            return RS.rank1(rs, block_end) - 1;
        }

        /// i-th child (0-indexed) of node `node_idx`. Returns NONE if out of range.
        pub fn childAt(data: []const u8, node_idx: IndexType, child_i: IndexType) IndexType {
            const d = degree(data, node_idx);
            if (child_i >= d) return NONE;

            const rs = rsData(data);
            const block_start = RS.select0(rs, node_idx) + 1;
            const one_pos = block_start + child_i;
            return RS.rank1(rs, one_pos + 1) - 1;
        }

        /// Next sibling of node `i`. Returns NONE if `i` is the last sibling.
        ///
        /// Node i's 1-bit is at select1(i) (0-indexed). If the next position
        /// is also a 1-bit, that's node i+1 (the next sibling in the same block).
        pub fn nextSibling(data: []const u8, i: IndexType) IndexType {
            const rs = rsData(data);
            const pos = RS.select1(rs, i);
            const next_pos = pos + 1;
            const bv_len = RS.totalBits(rs);

            if (next_pos >= bv_len) return NONE;
            if (RS.getBit(rs, next_pos) == 0) return NONE;

            return i + 1;
        }

        /// Previous sibling of node `i`. Returns NONE if `i` is the first sibling.
        pub fn prevSibling(data: []const u8, i: IndexType) IndexType {
            if (i == 0) return NONE;

            const rs = rsData(data);
            const pos = RS.select1(rs, i);
            if (pos == 0) return NONE;

            if (RS.getBit(rs, pos - 1) == 0) return NONE;
            return i - 1;
        }

        /// Parent of node `i`. Returns NONE for the root (node 0).
        ///
        /// Node i's 1-bit is at position p = select1(i) (0-indexed).
        /// rank0(p) counts 0-bits in [0, p), giving the entity number
        /// whose degree-block contains p.  Entity k = node (k-1).
        /// So parent = rank0(p) - 1, or NONE if rank0(p) == 0.
        pub fn parent(data: []const u8, i: IndexType) IndexType {
            if (i == 0) return NONE;

            const rs = rsData(data);
            const p = RS.select1(rs, i);
            const entity = RS.rank0(rs, p);
            if (entity == 0) return NONE;
            return entity - 1;
        }

        /// Check if node `i` is a leaf (has no children).
        pub fn isLeaf(data: []const u8, i: IndexType) bool {
            return degree(data, i) == 0;
        }

        /// Total number of nodes in the subtree rooted at `i` (including `i`).
        /// Warning: O(subtree size) — walks the subtree.
        pub fn subtreeSize(data: []const u8, i: IndexType) IndexType {
            var size: IndexType = 1;
            const d = degree(data, i);
            if (d == 0) return 1;

            const fc = firstChild(data, i);
            if (fc == NONE) return 1;

            var child = fc;
            var counted: IndexType = 0;
            while (child != NONE and counted < d) {
                size += subtreeSize(data, child);
                child = nextSibling(data, child);
                counted += 1;
            }
            return size;
        }

        /// Depth of node `i` (root has depth 0).
        /// O(depth) — walks up to root.
        pub fn depth(data: []const u8, i: IndexType) IndexType {
            var d: IndexType = 0;
            var node = i;
            while (true) {
                const p = parent(data, node);
                if (p == NONE) break;
                node = p;
                d += 1;
            }
            return d;
        }

        // ── Internal helpers ──

        fn readIdx(data: []const u8, offset: u32) IndexType {
            const bytes = data[offset..][0..index_size];
            return std.mem.readInt(IndexType, bytes, .little);
        }

        fn writeIdx(data: []u8, offset: u32, value: IndexType) void {
            const bytes = data[offset..][0..index_size];
            std.mem.writeInt(IndexType, bytes, value, .little);
        }
    };
}

// ═════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════

const testing = std.testing;

// Test tree:
//       A (root, degree 2)
//      / \
//     B   C
//    /|   |
//   D E   F
//
// BFS order: A, B, C, D, E, F
// Degrees:   2, 2, 1, 0, 0, 0

test "LOUDS: basic tree — degrees, children, parent" {
    const L = Louds(u32);
    const degrees = [_]u32{ 2, 2, 1, 0, 0, 0 };
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    // Number of nodes
    try testing.expectEqual(@as(u32, 6), L.numNodes(data));

    // Degrees
    try testing.expectEqual(@as(u32, 2), L.degree(data, 0)); // A
    try testing.expectEqual(@as(u32, 2), L.degree(data, 1)); // B
    try testing.expectEqual(@as(u32, 1), L.degree(data, 2)); // C
    try testing.expectEqual(@as(u32, 0), L.degree(data, 3)); // D
    try testing.expectEqual(@as(u32, 0), L.degree(data, 4)); // E
    try testing.expectEqual(@as(u32, 0), L.degree(data, 5)); // F

    // First child
    try testing.expectEqual(@as(u32, 1), L.firstChild(data, 0)); // A → B
    try testing.expectEqual(@as(u32, 3), L.firstChild(data, 1)); // B → D
    try testing.expectEqual(@as(u32, 5), L.firstChild(data, 2)); // C → F
    try testing.expectEqual(L.NONE, L.firstChild(data, 3)); // D is leaf
    try testing.expectEqual(L.NONE, L.firstChild(data, 4)); // E is leaf
    try testing.expectEqual(L.NONE, L.firstChild(data, 5)); // F is leaf

    // Last child
    try testing.expectEqual(@as(u32, 2), L.lastChild(data, 0)); // A → C
    try testing.expectEqual(@as(u32, 4), L.lastChild(data, 1)); // B → E
    try testing.expectEqual(@as(u32, 5), L.lastChild(data, 2)); // C → F

    // Parent
    try testing.expectEqual(L.NONE, L.parent(data, 0)); // A is root
    try testing.expectEqual(@as(u32, 0), L.parent(data, 1)); // B → A
    try testing.expectEqual(@as(u32, 0), L.parent(data, 2)); // C → A
    try testing.expectEqual(@as(u32, 1), L.parent(data, 3)); // D → B
    try testing.expectEqual(@as(u32, 1), L.parent(data, 4)); // E → B
    try testing.expectEqual(@as(u32, 2), L.parent(data, 5)); // F → C

    // Next sibling
    try testing.expectEqual(L.NONE, L.nextSibling(data, 0)); // root has no sibling
    try testing.expectEqual(@as(u32, 2), L.nextSibling(data, 1)); // B → C
    try testing.expectEqual(L.NONE, L.nextSibling(data, 2)); // C has no next sibling
    try testing.expectEqual(@as(u32, 4), L.nextSibling(data, 3)); // D → E
    try testing.expectEqual(L.NONE, L.nextSibling(data, 4)); // E has no next sibling

    // Is leaf
    try testing.expect(!L.isLeaf(data, 0));
    try testing.expect(!L.isLeaf(data, 1));
    try testing.expect(!L.isLeaf(data, 2));
    try testing.expect(L.isLeaf(data, 3));
    try testing.expect(L.isLeaf(data, 4));
    try testing.expect(L.isLeaf(data, 5));
}

test "LOUDS: single node tree (root only)" {
    const L = Louds(u32);
    const degrees = [_]u32{0};
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 1), L.numNodes(data));
    try testing.expectEqual(@as(u32, 0), L.degree(data, 0));
    try testing.expect(L.isLeaf(data, 0));
    try testing.expectEqual(L.NONE, L.firstChild(data, 0));
    try testing.expectEqual(L.NONE, L.parent(data, 0));
}

test "LOUDS: linear tree (each node has one child)" {
    // A → B → C → D (chain of 4)
    const L = Louds(u32);
    const degrees = [_]u32{ 1, 1, 1, 0 };
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 4), L.numNodes(data));

    // Each node (except last) has exactly one child
    try testing.expectEqual(@as(u32, 1), L.firstChild(data, 0));
    try testing.expectEqual(@as(u32, 2), L.firstChild(data, 1));
    try testing.expectEqual(@as(u32, 3), L.firstChild(data, 2));
    try testing.expectEqual(L.NONE, L.firstChild(data, 3));

    // Parents
    try testing.expectEqual(L.NONE, L.parent(data, 0));
    try testing.expectEqual(@as(u32, 0), L.parent(data, 1));
    try testing.expectEqual(@as(u32, 1), L.parent(data, 2));
    try testing.expectEqual(@as(u32, 2), L.parent(data, 3));

    // Depths
    try testing.expectEqual(@as(u32, 0), L.depth(data, 0));
    try testing.expectEqual(@as(u32, 1), L.depth(data, 1));
    try testing.expectEqual(@as(u32, 2), L.depth(data, 2));
    try testing.expectEqual(@as(u32, 3), L.depth(data, 3));
}

test "LOUDS: wide tree (root has many children)" {
    // Root has 5 children, all leaves
    const L = Louds(u32);
    const degrees = [_]u32{ 5, 0, 0, 0, 0, 0 };
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 6), L.numNodes(data));
    try testing.expectEqual(@as(u32, 5), L.degree(data, 0));

    // All children are leaves
    for (1..6) |i| {
        try testing.expect(L.isLeaf(data, @intCast(i)));
        try testing.expectEqual(@as(u32, 0), L.parent(data, @intCast(i)));
    }

    // Child at
    for (0..5) |i| {
        try testing.expectEqual(@as(u32, @intCast(i + 1)), L.childAt(data, 0, @intCast(i)));
    }
    try testing.expectEqual(L.NONE, L.childAt(data, 0, 5)); // out of range
}

test "LOUDS: subtree size" {
    // Tree: A(2) → B(2), C(1). B → D(0), E(0). C → F(0).
    const L = Louds(u32);
    const degrees = [_]u32{ 2, 2, 1, 0, 0, 0 };
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    try testing.expectEqual(@as(u32, 6), L.subtreeSize(data, 0)); // whole tree
    try testing.expectEqual(@as(u32, 3), L.subtreeSize(data, 1)); // B, D, E
    try testing.expectEqual(@as(u32, 2), L.subtreeSize(data, 2)); // C, F
    try testing.expectEqual(@as(u32, 1), L.subtreeSize(data, 3)); // D (leaf)
    try testing.expectEqual(@as(u32, 1), L.subtreeSize(data, 4)); // E (leaf)
    try testing.expectEqual(@as(u32, 1), L.subtreeSize(data, 5)); // F (leaf)
}

test "LOUDS: childAt random access" {
    // Tree: root(3) → A, B, C. A(2) → D, E.
    const L = Louds(u32);
    const degrees = [_]u32{ 3, 2, 0, 0, 0, 0 };
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    // Root's children
    try testing.expectEqual(@as(u32, 1), L.childAt(data, 0, 0)); // A
    try testing.expectEqual(@as(u32, 2), L.childAt(data, 0, 1)); // B
    try testing.expectEqual(@as(u32, 3), L.childAt(data, 0, 2)); // C

    // A's children
    try testing.expectEqual(@as(u32, 4), L.childAt(data, 1, 0)); // D
    try testing.expectEqual(@as(u32, 5), L.childAt(data, 1, 1)); // E
}

test "LOUDS: prevSibling" {
    const L = Louds(u32);
    const degrees = [_]u32{ 3, 0, 0, 0 };
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    // Root's children are nodes 1, 2, 3
    try testing.expectEqual(L.NONE, L.prevSibling(data, 1)); // first child
    try testing.expectEqual(@as(u32, 1), L.prevSibling(data, 2));
    try testing.expectEqual(@as(u32, 2), L.prevSibling(data, 3));
}

test "LOUDS: empty tree returns error" {
    const L = Louds(u32);
    const degrees = [_]u32{};
    try testing.expectError(error.EmptyTree, L.buildFromDegreesUniform(testing.allocator, &degrees));
}

test "LOUDS: invalid tree (degrees don't form a tree)" {
    const L = Louds(u32);
    // Degrees sum to 3, but n-1 = 1. Invalid.
    const degrees = [_]u32{ 3, 0 };
    try testing.expectError(error.InvalidTree, L.buildFromDegreesUniform(testing.allocator, &degrees));
}

test "LOUDS: space is approximately 2n+1 bits + overhead" {
    const L = Louds(u32);
    // Perfect binary tree of depth 3: 7 nodes
    const degrees = [_]u32{ 2, 2, 2, 0, 0, 0, 0 };
    const data = try L.buildFromDegreesUniform(testing.allocator, &degrees);
    defer testing.allocator.free(data);

    // 2*7 + 1 = 15 bits = 2 bytes for the raw bitvector
    // Plus rank/select overhead + header
    // Should be well under 200 bytes for 7 nodes
    try testing.expect(data.len < 200);
}
