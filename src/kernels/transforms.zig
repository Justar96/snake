const std = @import("std");
const simd = @import("../simd/simd.zig");
const Vec = simd.VecF64;
const VecLen = simd.VecF64Len;

// =============================================================================
// Transform Kernels (in-place mutations)
// =============================================================================

/// Clip values in-place to [lo, hi]
/// Uses SIMD for vectorized min/max operations
pub fn clip(ptr: [*]f64, len: usize, lo: f64, hi: f64) void {
    var i: usize = 0;
    const lo_vec: Vec = @splat(lo);
    const hi_vec: Vec = @splat(hi);

    // SIMD loop
    while (i + VecLen <= len) : (i += VecLen) {
        const v: Vec = ptr[i..][0..VecLen].*;
        ptr[i..][0..VecLen].* = @max(lo_vec, @min(hi_vec, v));
    }

    // Scalar tail
    while (i < len) : (i += 1) {
        ptr[i] = @max(lo, @min(hi, ptr[i]));
    }
}

/// L2 normalize in-place: x[i] /= sqrt(Σx²)
pub fn normalize(ptr: [*]f64, len: usize) void {
    if (len == 0) return;

    // Compute L2 norm using SIMD sum of squares
    const sum_sq = simd.sumSqSimd(ptr, len);
    if (sum_sq == 0.0) return;

    const inv_norm = 1.0 / @sqrt(sum_sq);

    // Scale by inverse norm using SIMD
    var i: usize = 0;
    const inv_vec: Vec = @splat(inv_norm);

    while (i + VecLen <= len) : (i += VecLen) {
        const v: Vec = ptr[i..][0..VecLen].*;
        ptr[i..][0..VecLen].* = v * inv_vec;
    }

    // Scalar tail
    while (i < len) : (i += 1) {
        ptr[i] *= inv_norm;
    }
}

/// Multiply array by scalar in-place: x[i] *= s
pub fn scale(ptr: [*]f64, len: usize, scalar: f64) void {
    var i: usize = 0;
    const s_vec: Vec = @splat(scalar);

    while (i + VecLen <= len) : (i += VecLen) {
        const v: Vec = ptr[i..][0..VecLen].*;
        ptr[i..][0..VecLen].* = v * s_vec;
    }

    while (i < len) : (i += 1) {
        ptr[i] *= scalar;
    }
}

/// SAXPY: a[i] += scalar * b[i]
pub fn saxpy(a_ptr: [*]f64, b_ptr: [*]const f64, len: usize, scalar: f64) void {
    var i: usize = 0;
    const s_vec: Vec = @splat(scalar);

    while (i + VecLen <= len) : (i += VecLen) {
        const va: Vec = a_ptr[i..][0..VecLen].*;
        const vb: Vec = b_ptr[i..][0..VecLen].*;
        a_ptr[i..][0..VecLen].* = va + s_vec * vb;
    }

    while (i < len) : (i += 1) {
        a_ptr[i] += scalar * b_ptr[i];
    }
}

// =============================================================================
// Tests
// =============================================================================

test "clip basic" {
    var data = [_]f64{ -1.0, 0.5, 1.5, 3.0 };
    clip(&data, data.len, 0.0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), data[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), data[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), data[3], 1e-10);
}

test "normalize basic" {
    var data = [_]f64{ 3.0, 4.0 };
    normalize(&data, data.len);
    // L2 norm = 5, so normalized = [0.6, 0.8]
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), data[1], 1e-10);
}

test "scale basic" {
    var data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    scale(&data, data.len, 2.0);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), data[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), data[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), data[3], 1e-10);
}

test "saxpy basic" {
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f64{ 1.0, 1.0, 1.0, 1.0 };
    saxpy(&a, &b, a.len, 2.0);
    // a = a + 2*b = [3, 4, 5, 6]
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), a[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), a[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), a[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), a[3], 1e-10);
}
