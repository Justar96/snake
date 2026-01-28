const std = @import("std");

// =============================================================================
// SIMD Type Aliases (local copies to avoid circular imports)
// =============================================================================

const Vec4 = @Vector(4, f64);

// =============================================================================
// SIMD Reduction Helpers
//
// These provide reusable reduction patterns using explicit SIMD vectorization.
// All functions follow the pattern: SIMD loop (4 elements) + scalar tail.
// =============================================================================

/// Sum of array elements: Σ(x)
pub fn sum(ptr: [*]const f64, len: usize) f64 {
    var acc: Vec4 = @splat(0.0);
    var i: usize = 0;

    // SIMD loop: process 4 elements at a time
    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = ptr[i..][0..4].*;
        acc += v;
    }

    // Reduce SIMD accumulator to scalar
    var result = @reduce(.Add, acc);

    // Scalar tail: remaining elements
    while (i < len) : (i += 1) {
        result += ptr[i];
    }

    return result;
}

/// Sum of squares: Σ(x²)
pub fn sumSq(ptr: [*]const f64, len: usize) f64 {
    var acc: Vec4 = @splat(0.0);
    var i: usize = 0;

    // SIMD loop: process 4 elements at a time
    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = ptr[i..][0..4].*;
        acc += v * v;
    }

    // Reduce SIMD accumulator to scalar
    var result = @reduce(.Add, acc);

    // Scalar tail: remaining elements
    while (i < len) : (i += 1) {
        result += ptr[i] * ptr[i];
    }

    return result;
}

/// Dot product: Σ(a · b)
pub fn dot(a_ptr: [*]const f64, b_ptr: [*]const f64, len: usize) f64 {
    var acc: Vec4 = @splat(0.0);
    var i: usize = 0;

    // SIMD loop: process 4 elements at a time
    while (i + 4 <= len) : (i += 4) {
        const va: Vec4 = a_ptr[i..][0..4].*;
        const vb: Vec4 = b_ptr[i..][0..4].*;
        acc += va * vb;
    }

    // Reduce SIMD accumulator to scalar
    var result = @reduce(.Add, acc);

    // Scalar tail: remaining elements
    while (i < len) : (i += 1) {
        result += a_ptr[i] * b_ptr[i];
    }

    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "sum basic" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const result = sum(&data, data.len);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result, 1e-10);
}

test "sum with tail" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const result = sum(&data, data.len);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), result, 1e-10);
}

test "sum empty" {
    const data = [_]f64{};
    const result = sum(&data, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result, 1e-10);
}

test "sumSq basic" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const result = sumSq(&data, data.len);
    // 1 + 4 + 9 + 16 = 30
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), result, 1e-10);
}

test "sumSq with tail" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const result = sumSq(&data, data.len);
    // 1 + 4 + 9 + 16 + 25 = 55
    try std.testing.expectApproxEqAbs(@as(f64, 55.0), result, 1e-10);
}

test "sumSq empty" {
    const data = [_]f64{};
    const result = sumSq(&data, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result, 1e-10);
}

test "dot basic" {
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f64{ 2.0, 3.0, 4.0, 5.0 };
    const result = dot(&a, &b, a.len);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    try std.testing.expectApproxEqAbs(@as(f64, 40.0), result, 1e-10);
}

test "dot with tail" {
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const b = [_]f64{ 2.0, 3.0, 4.0, 5.0, 6.0 };
    const result = dot(&a, &b, a.len);
    // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
    try std.testing.expectApproxEqAbs(@as(f64, 70.0), result, 1e-10);
}

test "dot empty" {
    const a = [_]f64{};
    const b = [_]f64{};
    const result = dot(&a, &b, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result, 1e-10);
}
