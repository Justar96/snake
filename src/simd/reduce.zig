const std = @import("std");

// =============================================================================
// SIMD Type Aliases (local copies to avoid circular imports)
// =============================================================================

const VecLen: usize = std.simd.suggestVectorLength(f64) orelse 1;
const Vec = @Vector(VecLen, f64);

// Prefetch distance in elements (~2KB ahead = 256 f64 values)
const PREFETCH_DISTANCE: usize = 256;

// =============================================================================
// SIMD Reduction Helpers
//
// These provide reusable reduction patterns using explicit SIMD vectorization.
// All functions follow the pattern: SIMD loop (VecLen elements) + scalar tail.
// =============================================================================

/// Sum of array elements: Σ(x)
pub fn sum(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0.0;

    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var acc2: Vec = @splat(0.0);
    var acc3: Vec = @splat(0.0);
    var i: usize = 0;
    const step = VecLen * 4;

    // SIMD loop: unroll by 4 to improve ILP.
    while (i + step <= len) : (i += step) {
        if (i + PREFETCH_DISTANCE < len) {
            @prefetch(ptr + i + PREFETCH_DISTANCE, .{ .locality = 3, .cache = .data });
        }
        const v0: Vec = ptr[i..][0..VecLen].*;
        const v1: Vec = ptr[i + VecLen ..][0..VecLen].*;
        const v2: Vec = ptr[i + VecLen * 2 ..][0..VecLen].*;
        const v3: Vec = ptr[i + VecLen * 3 ..][0..VecLen].*;
        acc0 += v0;
        acc1 += v1;
        acc2 += v2;
        acc3 += v3;
    }

    var acc: Vec = (acc0 + acc1) + (acc2 + acc3);
    while (i + VecLen <= len) : (i += VecLen) {
        const v: Vec = ptr[i..][0..VecLen].*;
        acc += v;
    }

    var result = @reduce(.Add, acc);

    // Scalar tail: remaining elements
    while (i < len) : (i += 1) {
        result += ptr[i];
    }

    return result;
}

/// Sum of squares: Σ(x²)
pub fn sumSq(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0.0;

    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var acc2: Vec = @splat(0.0);
    var acc3: Vec = @splat(0.0);
    var i: usize = 0;
    const step = VecLen * 4;

    // SIMD loop: unroll by 4 to improve ILP and enable FMA.
    while (i + step <= len) : (i += step) {
        if (i + PREFETCH_DISTANCE < len) {
            @prefetch(ptr + i + PREFETCH_DISTANCE, .{ .locality = 3, .cache = .data });
        }
        const v0: Vec = ptr[i..][0..VecLen].*;
        const v1: Vec = ptr[i + VecLen ..][0..VecLen].*;
        const v2: Vec = ptr[i + VecLen * 2 ..][0..VecLen].*;
        const v3: Vec = ptr[i + VecLen * 3 ..][0..VecLen].*;
        acc0 = @mulAdd(Vec, v0, v0, acc0);
        acc1 = @mulAdd(Vec, v1, v1, acc1);
        acc2 = @mulAdd(Vec, v2, v2, acc2);
        acc3 = @mulAdd(Vec, v3, v3, acc3);
    }

    var acc: Vec = (acc0 + acc1) + (acc2 + acc3);
    while (i + VecLen <= len) : (i += VecLen) {
        const v: Vec = ptr[i..][0..VecLen].*;
        acc = @mulAdd(Vec, v, v, acc);
    }

    var result = @reduce(.Add, acc);

    // Scalar tail: remaining elements
    while (i < len) : (i += 1) {
        const x = ptr[i];
        result += x * x;
    }

    return result;
}

/// Dot product: Σ(a · b)
pub fn dot(a_ptr: [*]const f64, b_ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0.0;

    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var acc2: Vec = @splat(0.0);
    var acc3: Vec = @splat(0.0);
    var i: usize = 0;
    const step = VecLen * 4;

    // SIMD loop: unroll by 4 to improve ILP and enable FMA.
    while (i + step <= len) : (i += step) {
        if (i + PREFETCH_DISTANCE < len) {
            @prefetch(a_ptr + i + PREFETCH_DISTANCE, .{ .locality = 3, .cache = .data });
            @prefetch(b_ptr + i + PREFETCH_DISTANCE, .{ .locality = 3, .cache = .data });
        }
        const va0: Vec = a_ptr[i..][0..VecLen].*;
        const vb0: Vec = b_ptr[i..][0..VecLen].*;
        const va1: Vec = a_ptr[i + VecLen ..][0..VecLen].*;
        const vb1: Vec = b_ptr[i + VecLen ..][0..VecLen].*;
        const va2: Vec = a_ptr[i + VecLen * 2 ..][0..VecLen].*;
        const vb2: Vec = b_ptr[i + VecLen * 2 ..][0..VecLen].*;
        const va3: Vec = a_ptr[i + VecLen * 3 ..][0..VecLen].*;
        const vb3: Vec = b_ptr[i + VecLen * 3 ..][0..VecLen].*;
        acc0 = @mulAdd(Vec, va0, vb0, acc0);
        acc1 = @mulAdd(Vec, va1, vb1, acc1);
        acc2 = @mulAdd(Vec, va2, vb2, acc2);
        acc3 = @mulAdd(Vec, va3, vb3, acc3);
    }

    var acc: Vec = (acc0 + acc1) + (acc2 + acc3);
    while (i + VecLen <= len) : (i += VecLen) {
        const va: Vec = a_ptr[i..][0..VecLen].*;
        const vb: Vec = b_ptr[i..][0..VecLen].*;
        acc = @mulAdd(Vec, va, vb, acc);
    }

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
