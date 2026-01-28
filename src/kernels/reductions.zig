const std = @import("std");
const simd = @import("../simd/simd.zig");
const reduce = simd.reduce;
const Vec = simd.VecF64;
const VecLen = simd.VecF64Len;

// =============================================================================
// Reduction Kernels
//
// These kernels delegate to the SIMD reduce helpers for core operations.
// =============================================================================

/// Sum of squares: Σ(x²)
/// Uses explicit SIMD for guaranteed vectorization.
pub fn sumSq(ptr: [*]const f64, len: usize) f64 {
    return reduce.sumSq(ptr, len);
}

/// Dot product: Σ(a · b)
/// Uses explicit SIMD for guaranteed vectorization.
pub fn dot(a_ptr: [*]const f64, b_ptr: [*]const f64, len: usize) f64 {
    return reduce.dot(a_ptr, b_ptr, len);
}

/// Find index of maximum value
/// Single-pass SIMD: tracks both max values AND indices in parallel using @select.
/// More cache-friendly than two-pass approach.
pub fn argmax(ptr: [*]const f64, len: usize) usize {
    if (len == 0) return 0;
    if (len == 1) return 0;

    // Handle small arrays without SIMD
    if (len < VecLen) {
        var max_idx: usize = 0;
        var max_val: f64 = ptr[0];
        for (1..len) |idx| {
            if (ptr[idx] > max_val) {
                max_val = ptr[idx];
                max_idx = idx;
            }
        }
        return max_idx;
    }

    const IdxVec = @Vector(VecLen, usize);

    // Initialize with first VecLen elements
    var max_vec: Vec = ptr[0..VecLen].*;
    var idx_vec: IdxVec = undefined;
    inline for (0..VecLen) |lane| {
        idx_vec[lane] = lane;
    }
    var i: usize = VecLen;

    // SIMD loop: track both max values and indices in parallel
    while (i + VecLen <= len) : (i += VecLen) {
        const v: Vec = ptr[i..][0..VecLen].*;
        var new_idx: IdxVec = undefined;
        inline for (0..VecLen) |lane| {
            new_idx[lane] = i + lane;
        }

        // Compare: which lanes have larger values?
        const mask = v > max_vec;

        // Select new values/indices where v > max_vec
        max_vec = @select(f64, mask, v, max_vec);
        idx_vec = @select(usize, mask, new_idx, idx_vec);
    }

    // Horizontal reduction: find the lane with the maximum
    var max_val = max_vec[0];
    var max_idx = idx_vec[0];
    inline for (1..VecLen) |lane| {
        if (max_vec[lane] > max_val) {
            max_val = max_vec[lane];
            max_idx = idx_vec[lane];
        }
    }

    // Handle tail elements
    while (i < len) : (i += 1) {
        if (ptr[i] > max_val) {
            max_val = ptr[i];
            max_idx = i;
        }
    }

    return max_idx;
}

/// Variance using two-pass algorithm with SIMD for performance
/// First pass: compute mean, Second pass: compute sum of squared differences
/// Returns population variance (divide by N, not N-1)
pub fn variance(ptr: [*]const f64, len: usize) f64 {
    if (len == 0) return 0.0;

    const n_f64 = @as(f64, @floatFromInt(len));

    // Pass 1: Compute mean using SIMD sum helper
    const mean = reduce.sum(ptr, len) / n_f64;

    // Pass 2: Compute sum of squared differences using SIMD
    var i: usize = 0;
    var sq_diff_vec: Vec = @splat(0.0);
    const mean_vec: Vec = @splat(mean);
    while (i + VecLen <= len) : (i += VecLen) {
        const v: Vec = ptr[i..][0..VecLen].*;
        const diff = v - mean_vec;
        sq_diff_vec += diff * diff;
    }
    var sq_diff_sum = @reduce(.Add, sq_diff_vec);
    // Scalar tail for squared differences
    while (i < len) : (i += 1) {
        const diff = ptr[i] - mean;
        sq_diff_sum += diff * diff;
    }

    return sq_diff_sum / n_f64;
}

// =============================================================================
// Tests
// =============================================================================

test "sumSq basic" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const result = sumSq(&data, data.len);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), result, 1e-10);
}

test "dot basic" {
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f64{ 2.0, 3.0, 4.0, 5.0 };
    const result = dot(&a, &b, a.len);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    try std.testing.expectApproxEqAbs(@as(f64, 40.0), result, 1e-10);
}

test "argmax basic" {
    const data = [_]f64{ 1.0, 4.0, 2.0, 3.0 };
    const result = argmax(&data, data.len);
    try std.testing.expectEqual(@as(usize, 1), result);
}

test "argmax duplicate max returns first occurrence" {
    const data = [_]f64{ 1.0, 5.0, 3.0, 5.0, 2.0 };
    const result = argmax(&data, data.len);
    try std.testing.expectEqual(@as(usize, 1), result); // first 5.0 is at index 1
}

test "variance basic" {
    const data = [_]f64{ 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 };
    const result = variance(&data, data.len);
    // Mean = 5, variance = 4
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result, 1e-10);
}
