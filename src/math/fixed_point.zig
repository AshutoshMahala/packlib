//! Deterministic fixed-point arithmetic types.
//!
//! Provides `FixedPoint(IntBits, FracBits)` for cross-platform reproducible math.
//! All operations are integer-backed — no floating point used in computation.
//!
//! ```
//! const Q = packlib.Q16_16;
//! const half = Q.fromFraction(1, 2);
//! const quarter = half.mul(half);  // 0.25 exactly
//! ```

pub fn FixedPoint(comptime IntBits: u8, comptime FracBits: u8) type {
    const total_bits = IntBits + FracBits;
    const BackingType = @Type(.{ .int = .{
        .signedness = .signed,
        .bits = total_bits,
    } });
    const WideType = @Type(.{ .int = .{
        .signedness = .signed,
        .bits = total_bits * 2,
    } });
    const IntType = @Type(.{ .int = .{
        .signedness = .signed,
        .bits = IntBits,
    } });

    return struct {
        const Self = @This();
        pub const backing_type = BackingType;
        pub const wide_type = WideType;
        pub const int_type = IntType;
        pub const int_bits = IntBits;
        pub const frac_bits = FracBits;

        raw: BackingType,

        pub const zero: Self = .{ .raw = 0 };
        pub const one: Self = .{ .raw = @as(BackingType, 1) << FracBits };

        // ── Construction ──

        pub fn fromInt(val: IntType) Self {
            return .{ .raw = @as(BackingType, val) << FracBits };
        }

        /// Construct from a rational number `numerator / denominator`.
        /// Uses widening arithmetic to preserve precision.
        /// Returns error on division by zero.
        pub fn fromFraction(numerator: i32, denominator: i32) !Self {
            if (denominator == 0) return error.DivisionByZero;
            // (numerator << FracBits) / denominator
            const wide_num: WideType = @as(WideType, numerator) << FracBits;
            const result = @divTrunc(wide_num, @as(WideType, denominator));
            return .{ .raw = @intCast(result) };
        }

        pub fn fromRaw(raw: BackingType) Self {
            return .{ .raw = raw };
        }

        // ── Arithmetic (wrapping) ──

        pub fn add(self: Self, other: Self) Self {
            return .{ .raw = self.raw +% other.raw };
        }

        pub fn sub(self: Self, other: Self) Self {
            return .{ .raw = self.raw -% other.raw };
        }

        /// Multiply using widening to preserve precision. Truncates toward zero.
        pub fn mul(self: Self, other: Self) Self {
            const wide_result = @as(WideType, self.raw) * @as(WideType, other.raw);
            return .{ .raw = @intCast(wide_result >> FracBits) };
        }

        /// Divide using widening to preserve precision. Truncates toward zero.
        /// Returns error on division by zero.
        pub fn div(self: Self, other: Self) !Self {
            if (other.raw == 0) return error.DivisionByZero;
            const wide_self = @as(WideType, self.raw) << FracBits;
            return .{ .raw = @intCast(@divTrunc(wide_self, @as(WideType, other.raw))) };
        }

        // ── Checked arithmetic ──

        /// Checked addition — returns error on overflow.
        pub fn checkedAdd(self: Self, other: Self) !Self {
            const result = @addWithOverflow(self.raw, other.raw);
            if (result[1] != 0) return error.Overflow;
            return .{ .raw = result[0] };
        }

        /// Checked subtraction — returns error on overflow.
        pub fn checkedSub(self: Self, other: Self) !Self {
            const result = @subWithOverflow(self.raw, other.raw);
            if (result[1] != 0) return error.Overflow;
            return .{ .raw = result[0] };
        }

        // ── Comparison ──

        pub fn lessThan(self: Self, other: Self) bool {
            return self.raw < other.raw;
        }

        pub fn lessThanOrEqual(self: Self, other: Self) bool {
            return self.raw <= other.raw;
        }

        pub fn greaterThan(self: Self, other: Self) bool {
            return self.raw > other.raw;
        }

        pub fn equal(self: Self, other: Self) bool {
            return self.raw == other.raw;
        }

        // ── Conversion ──

        /// Convert to f64 (for display/diagnostics only — not deterministic across platforms).
        pub fn toFloat(self: Self) f64 {
            return @as(f64, @floatFromInt(self.raw)) / @as(f64, @floatFromInt(@as(BackingType, 1) << FracBits));
        }

        /// Truncate fractional part and return the integer portion.
        pub fn toInt(self: Self) IntType {
            return @intCast(self.raw >> FracBits);
        }

        /// Absolute value.
        pub fn abs(self: Self) Self {
            if (self.raw < 0) {
                return .{ .raw = -%self.raw };
            }
            return self;
        }

        /// Negate.
        pub fn negate(self: Self) Self {
            return .{ .raw = -%self.raw };
        }

        /// Return the fractional part as a raw value (for inspection/debugging).
        pub fn fractionalRaw(self: Self) BackingType {
            const mask: BackingType = (@as(BackingType, 1) << FracBits) - 1;
            return self.raw & mask;
        }
    };
}

pub const Q16_16 = FixedPoint(16, 16);
pub const Q32_32 = FixedPoint(32, 32);

// ── Tests ────────────────────────────────────────────────────
const testing = @import("std").testing;

test "Q16_16: fromInt round-trip" {
    const q = Q16_16.fromInt(42);
    try testing.expectEqual(@as(i16, 42), q.toInt());
}

test "Q16_16: fromInt negative" {
    const q = Q16_16.fromInt(-7);
    try testing.expectEqual(@as(i16, -7), q.toInt());
}

test "Q16_16: add" {
    const a = Q16_16.fromInt(10);
    const b = Q16_16.fromInt(20);
    try testing.expectEqual(@as(i16, 30), a.add(b).toInt());
}

test "Q16_16: sub" {
    const a = Q16_16.fromInt(30);
    const b = Q16_16.fromInt(12);
    try testing.expectEqual(@as(i16, 18), a.sub(b).toInt());
}

test "Q16_16: fromFraction accuracy" {
    // 1/3 ≈ 0.333... → Q16_16 raw should be close to 65536/3 = 21845
    const third = try Q16_16.fromFraction(1, 3);
    const float_val = third.toFloat();
    // Should be within 1 ULP of the fractional precision: 1/65536 ≈ 0.0000153
    try testing.expect(@abs(float_val - 0.333333333) < 0.0001);
}

test "Q16_16: fromFraction 1/2" {
    const half = try Q16_16.fromFraction(1, 2);
    try testing.expectEqual(@as(i16, 0), half.toInt()); // integer part is 0
    const float_val = half.toFloat();
    try testing.expect(@abs(float_val - 0.5) < 0.0001);
}

test "Q16_16: fromFraction division by zero" {
    const result = Q16_16.fromFraction(1, 0);
    try testing.expectError(error.DivisionByZero, result);
}

test "Q16_16: mul precision — 0.5 * 0.5 = 0.25" {
    const half = try Q16_16.fromFraction(1, 2);
    const quarter = half.mul(half);
    try testing.expect(@abs(quarter.toFloat() - 0.25) < 0.0001);
}

test "Q16_16: mul integers" {
    const a = Q16_16.fromInt(6);
    const b = Q16_16.fromInt(7);
    try testing.expectEqual(@as(i16, 42), a.mul(b).toInt());
}

test "Q16_16: div precision" {
    const one = Q16_16.fromInt(1);
    const three = Q16_16.fromInt(3);
    const result = try one.div(three);
    try testing.expect(@abs(result.toFloat() - 0.333333333) < 0.0001);
}

test "Q16_16: div by zero" {
    const one = Q16_16.fromInt(1);
    const z = Q16_16.zero;
    try testing.expectError(error.DivisionByZero, one.div(z));
}

test "Q16_16: wrapping overflow" {
    const max_val = Q16_16.fromInt(32767);
    const one_val = Q16_16.fromInt(1);
    const wrapped = max_val.add(one_val);
    // Should wrap — the result won't be 32768
    try testing.expect(wrapped.toInt() != 32767 + 1);
}

test "Q16_16: checkedAdd overflow" {
    const max_val = Q16_16.fromInt(32767);
    const one_val = Q16_16.fromInt(1);
    try testing.expectError(error.Overflow, max_val.checkedAdd(one_val));
}

test "Q16_16: checkedAdd success" {
    const a = Q16_16.fromInt(10);
    const b = Q16_16.fromInt(20);
    const result = try a.checkedAdd(b);
    try testing.expectEqual(@as(i16, 30), result.toInt());
}

test "Q16_16: comparisons" {
    const a = Q16_16.fromInt(5);
    const b = Q16_16.fromInt(10);
    try testing.expect(a.lessThan(b));
    try testing.expect(!b.lessThan(a));
    try testing.expect(b.greaterThan(a));
    try testing.expect(a.equal(a));
    try testing.expect(!a.equal(b));
}

test "Q16_16: abs and negate" {
    const neg = Q16_16.fromInt(-5);
    try testing.expectEqual(@as(i16, 5), neg.abs().toInt());
    try testing.expectEqual(@as(i16, 5), neg.negate().toInt());

    const pos = Q16_16.fromInt(3);
    try testing.expectEqual(@as(i16, 3), pos.abs().toInt());
    try testing.expectEqual(@as(i16, -3), pos.negate().toInt());
}

test "Q16_16: zero and one constants" {
    try testing.expectEqual(@as(i16, 0), Q16_16.zero.toInt());
    try testing.expectEqual(@as(i16, 1), Q16_16.one.toInt());
    try testing.expect(@abs(Q16_16.one.toFloat() - 1.0) < 0.0001);
}

test "Q16_16: determinism — same inputs same raw bits" {
    const a1 = try Q16_16.fromFraction(355, 113); // pi approximation
    const a2 = try Q16_16.fromFraction(355, 113);
    try testing.expectEqual(a1.raw, a2.raw);

    const b1 = a1.mul(a1);
    const b2 = a2.mul(a2);
    try testing.expectEqual(b1.raw, b2.raw);
}

test "Q32_32: fromInt round-trip" {
    const q = Q32_32.fromInt(1000000);
    try testing.expectEqual(@as(i32, 1000000), q.toInt());
}

test "Q32_32: mul precision" {
    const half = try Q32_32.fromFraction(1, 2);
    const quarter = half.mul(half);
    try testing.expect(@abs(quarter.toFloat() - 0.25) < 0.00000001);
}

test "Cross-type consistency: Q16_16 and Q32_32 agree within Q16_16 precision" {
    const q16_third = try Q16_16.fromFraction(1, 3);
    const q32_third = try Q32_32.fromFraction(1, 3);

    const diff = @abs(q16_third.toFloat() - q32_third.toFloat());
    // Should agree within Q16_16 precision (1/65536 ≈ 0.0000153)
    try testing.expect(diff < 0.0001);
}
