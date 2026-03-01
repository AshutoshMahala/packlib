//! Tiered arena allocator configuration for multi-phase build pipelines.
//!
//! Compression and succinct data structure builds often proceed in phases:
//!
//!   1. **Temporary** — scratch space for intermediate computation (suffix arrays,
//!      heaps, hash maps).  Freed after each phase completes.
//!   2. **Intermediate** — ping-pong buffers for multi-pass algorithms (BWT output
//!      before MTF, trie before minimization).  Freed between passes.
//!   3. **Output** — the final serialized bytes.  Caller owns the result and
//!      controls its lifetime.
//!
//! `ArenaConfig` lets the user supply a separate allocator for each tier so that
//! short-lived allocations never fragment the long-lived output heap.
//!
//! ## Quick start
//!
//! ```zig
//! // Simple: same allocator for everything
//! const cfg = ArenaConfig.uniform(gpa);
//!
//! // Advanced: three separate arenas
//! var temp  = std.heap.ArenaAllocator.init(page_allocator);
//! var inter = std.heap.ArenaAllocator.init(page_allocator);
//! defer temp.deinit();
//! defer inter.deinit();
//!
//! const cfg = ArenaConfig{
//!     .temp = temp.allocator(),
//!     .intermediate = inter.allocator(),
//!     .output = user_allocator,
//! };
//! ```

const std = @import("std");

/// Tiered allocator configuration for build pipelines.
///
/// **Level 1 – `temp`**
///   Scratch space freed after each phase.  Used for suffix-array construction,
///   priority queues, frequency tables, and similar throwaway state.
///
/// **Level 2 – `intermediate`**
///   Ping-pong buffers for multi-pass algorithms.  After each pass the "from"
///   arena is reset and the roles are swapped.  Used for BWT ↔ MTF ↔ RLE
///   pipeline stages, trie ↔ minimized DAFSA, etc.
///
/// **Level 3 – `output`**
///   Long-lived allocator whose memory is returned to the caller.  Typically
///   backed by the user's own arena, GPA, or page allocator.
pub const ArenaConfig = struct {
    /// Level 1 — temporary scratch space.  Freed after each build phase.
    temp: std.mem.Allocator,

    /// Level 2 — intermediate ping-pong buffers.  Freed between passes.
    intermediate: std.mem.Allocator,

    /// Level 3 — long-lived output.  Caller owns the resulting memory.
    output: std.mem.Allocator,

    /// Create a config that uses the same allocator for all three tiers.
    ///
    /// This is the simplest option and behaves identically to passing a bare
    /// `std.mem.Allocator` to a build function.
    pub fn uniform(allocator: std.mem.Allocator) ArenaConfig {
        return .{
            .temp = allocator,
            .intermediate = allocator,
            .output = allocator,
        };
    }

    /// Create a two-tier config: one allocator for scratch work (temp +
    /// intermediate) and another for the final output the caller keeps.
    pub fn twoTier(scratch: std.mem.Allocator, output: std.mem.Allocator) ArenaConfig {
        return .{
            .temp = scratch,
            .intermediate = scratch,
            .output = output,
        };
    }

    /// Return just the output allocator, for legacy / Tier-1 compatibility
    /// where a single `std.mem.Allocator` is expected.
    pub fn outputAllocator(self: ArenaConfig) std.mem.Allocator {
        return self.output;
    }
};

// ── Tests ────────────────────────────────────────────────────

test "ArenaConfig.uniform: all three tiers share one allocator" {
    const a = std.testing.allocator;
    const cfg = ArenaConfig.uniform(a);
    // All three must resolve to the same allocator pointer.
    try std.testing.expect(cfg.temp.ptr == cfg.intermediate.ptr);
    try std.testing.expect(cfg.intermediate.ptr == cfg.output.ptr);
}

test "ArenaConfig.twoTier: scratch vs output distinction" {
    const a = std.testing.allocator;

    // Use two distinguishable allocators.
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const cfg = ArenaConfig.twoTier(arena.allocator(), a);
    try std.testing.expect(cfg.temp.ptr == cfg.intermediate.ptr);
    try std.testing.expect(cfg.temp.ptr != cfg.output.ptr);
}

test "ArenaConfig.outputAllocator returns L3" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const cfg = ArenaConfig{
        .temp = arena.allocator(),
        .intermediate = arena.allocator(),
        .output = a,
    };
    try std.testing.expect(cfg.outputAllocator().ptr == a.ptr);
}

test "ArenaConfig: allocate + free through each tier" {
    const a = std.testing.allocator;
    const cfg = ArenaConfig.uniform(a);

    // Temp allocation
    const tmp = try cfg.temp.alloc(u8, 64);
    defer cfg.temp.free(tmp);
    tmp[0] = 0xAA;
    try std.testing.expectEqual(@as(u8, 0xAA), tmp[0]);

    // Intermediate allocation
    const inter = try cfg.intermediate.alloc(u8, 128);
    defer cfg.intermediate.free(inter);
    inter[0] = 0xBB;
    try std.testing.expectEqual(@as(u8, 0xBB), inter[0]);

    // Output allocation
    const out = try cfg.output.alloc(u8, 256);
    defer cfg.output.free(out);
    out[0] = 0xCC;
    try std.testing.expectEqual(@as(u8, 0xCC), out[0]);
}
