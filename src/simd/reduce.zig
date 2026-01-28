const std = @import("std");

// =============================================================================
// SIMD Type Aliases (local copies to avoid circular imports)
// =============================================================================

const VecLen: usize = std.simd.suggestVectorLength(f64) orelse 1;
const Vec = @Vector(VecLen, f64);
const VecAlign: usize = @alignOf(Vec);

fn alignPrefix(ptr: [*]const f64, len: usize) usize {
    if (len == 0) return 0;
    if (VecAlign <= @alignOf(f64)) return 0;
    const mask = VecAlign - 1;
    const mis = @intFromPtr(ptr) & mask;
    if (mis == 0) return 0;
    const bytes = VecAlign - mis;
    const elems = bytes / @sizeOf(f64);
    return if (elems < len) elems else len;
}

fn alignPrefixPair(a_ptr: [*]const f64, b_ptr: [*]const f64, len: usize) usize {
    if (len == 0) return 0;
    if (VecAlign <= @alignOf(f64)) return 0;
    const mask = VecAlign - 1;
    const mis_a = @intFromPtr(a_ptr) & mask;
    const mis_b = @intFromPtr(b_ptr) & mask;
    if (mis_a != mis_b or mis_a == 0) return 0;
    const bytes = VecAlign - mis_a;
    const elems = bytes / @sizeOf(f64);
    return if (elems < len) elems else len;
}

// =============================================================================
// SIMD Reduction Helpers
//
// These provide reusable reduction patterns using explicit SIMD vectorization.
// All functions follow the pattern: SIMD loop (VecLen elements) + scalar tail.
// =============================================================================

/// Sum of array elements: Σ(x)
pub fn sum(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0.0;

    var result: f64 = 0.0;
    var i: usize = 0;
    const prefix = alignPrefix(ptr, len);
    while (i < prefix) : (i += 1) {
        result += ptr[i];
    }
    if (i >= len) return result;

    const aligned_len = len - i;
    const base: [*]align(VecAlign) const f64 = @alignCast(ptr + i);

    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var j: usize = 0;
    const step = VecLen * 2;

    // SIMD loop: unroll by 2 to improve ILP.
    while (j + step <= aligned_len) : (j += step) {
        const v0: Vec = base[j..][0..VecLen].*;
        const v1: Vec = base[j + VecLen ..][0..VecLen].*;
        acc0 += v0;
        acc1 += v1;
    }

    var acc: Vec = acc0 + acc1;
    while (j + VecLen <= aligned_len) : (j += VecLen) {
        const v: Vec = base[j..][0..VecLen].*;
        acc += v;
    }

    result += @reduce(.Add, acc);

    // Scalar tail: remaining elements
    while (j < aligned_len) : (j += 1) {
        result += base[j];
    }

    return result;
}

/// Sum of squares: Σ(x²)
pub fn sumSq(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0.0;

    var result: f64 = 0.0;
    var i: usize = 0;
    const prefix = alignPrefix(ptr, len);
    while (i < prefix) : (i += 1) {
        const x = ptr[i];
        result += x * x;
    }
    if (i >= len) return result;

    const aligned_len = len - i;
    const base: [*]align(VecAlign) const f64 = @alignCast(ptr + i);

    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var j: usize = 0;
    const step = VecLen * 2;

    // SIMD loop: unroll by 2 to improve ILP and enable FMA.
    while (j + step <= aligned_len) : (j += step) {
        const v0: Vec = base[j..][0..VecLen].*;
        const v1: Vec = base[j + VecLen ..][0..VecLen].*;
        acc0 = @mulAdd(Vec, v0, v0, acc0);
        acc1 = @mulAdd(Vec, v1, v1, acc1);
    }

    var acc: Vec = acc0 + acc1;
    while (j + VecLen <= aligned_len) : (j += VecLen) {
        const v: Vec = base[j..][0..VecLen].*;
        acc = @mulAdd(Vec, v, v, acc);
    }

    result += @reduce(.Add, acc);

    // Scalar tail: remaining elements
    while (j < aligned_len) : (j += 1) {
        const x = base[j];
        result += x * x;
    }

    return result;
}

/// Dot product: Σ(a · b)
pub fn dot(a_ptr: [*]const f64, b_ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0.0;

    var result: f64 = 0.0;
    var i: usize = 0;
    const prefix = alignPrefixPair(a_ptr, b_ptr, len);
    while (i < prefix) : (i += 1) {
        result += a_ptr[i] * b_ptr[i];
    }
    if (i >= len) return result;

    const remaining = len - i;
    const a_aligned = (@intFromPtr(a_ptr + i) & (VecAlign - 1)) == 0;
    const b_aligned = (@intFromPtr(b_ptr + i) & (VecAlign - 1)) == 0;

    if (a_aligned and b_aligned) {
        const a_base: [*]align(VecAlign) const f64 = @alignCast(a_ptr + i);
        const b_base: [*]align(VecAlign) const f64 = @alignCast(b_ptr + i);
        var acc0: Vec = @splat(0.0);
        var acc1: Vec = @splat(0.0);
        var j: usize = 0;
        const step = VecLen * 2;

        while (j + step <= remaining) : (j += step) {
            const va0: Vec = a_base[j..][0..VecLen].*;
            const vb0: Vec = b_base[j..][0..VecLen].*;
            const va1: Vec = a_base[j + VecLen ..][0..VecLen].*;
            const vb1: Vec = b_base[j + VecLen ..][0..VecLen].*;
            acc0 = @mulAdd(Vec, va0, vb0, acc0);
            acc1 = @mulAdd(Vec, va1, vb1, acc1);
        }

        var acc: Vec = acc0 + acc1;
        while (j + VecLen <= remaining) : (j += VecLen) {
            const va: Vec = a_base[j..][0..VecLen].*;
            const vb: Vec = b_base[j..][0..VecLen].*;
            acc = @mulAdd(Vec, va, vb, acc);
        }

        result += @reduce(.Add, acc);
        while (j < remaining) : (j += 1) {
            result += a_base[j] * b_base[j];
        }

        return result;
    }

    // Unaligned path.
    var acc0: Vec = @splat(0.0);
    var acc1: Vec = @splat(0.0);
    var j: usize = 0;
    const step = VecLen * 2;

    while (j + step <= remaining) : (j += step) {
        const va0: Vec = a_ptr[i + j ..][0..VecLen].*;
        const vb0: Vec = b_ptr[i + j ..][0..VecLen].*;
        const va1: Vec = a_ptr[i + j + VecLen ..][0..VecLen].*;
        const vb1: Vec = b_ptr[i + j + VecLen ..][0..VecLen].*;
        acc0 = @mulAdd(Vec, va0, vb0, acc0);
        acc1 = @mulAdd(Vec, va1, vb1, acc1);
    }

    var acc: Vec = acc0 + acc1;
    while (j + VecLen <= remaining) : (j += VecLen) {
        const va: Vec = a_ptr[i + j ..][0..VecLen].*;
        const vb: Vec = b_ptr[i + j ..][0..VecLen].*;
        acc = @mulAdd(Vec, va, vb, acc);
    }

    result += @reduce(.Add, acc);
    while (j < remaining) : (j += 1) {
        result += a_ptr[i + j] * b_ptr[i + j];
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
