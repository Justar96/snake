const std = @import("std");

// =============================================================================
// SIMD Configuration
// =============================================================================

const Vec4 = @Vector(4, f64);

// =============================================================================
// Core Kernels - Single Threaded
// =============================================================================

/// Sum of squares: Σ(x²)
/// Uses explicit SIMD for guaranteed vectorization.
pub export fn sum_sq_f64(ptr: [*]const f64, len: usize) f64 {
    return sumSqSimd(ptr, len);
}

fn sumSqSimd(ptr: [*]const f64, len: usize) f64 {
    var acc: Vec4 = @splat(0.0);
    var i: usize = 0;

    // SIMD loop: process 4 elements at a time
    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = ptr[i..][0..4].*;
        acc += v * v;
    }

    // Reduce SIMD accumulator
    var sum = @reduce(.Add, acc);

    // Scalar tail: remaining elements
    while (i < len) : (i += 1) {
        sum += ptr[i] * ptr[i];
    }

    return sum;
}

/// Dot product: Σ(a · b)
pub export fn dot_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, len: usize) f64 {
    var acc: Vec4 = @splat(0.0);
    var i: usize = 0;

    while (i + 4 <= len) : (i += 4) {
        const va: Vec4 = a_ptr[i..][0..4].*;
        const vb: Vec4 = b_ptr[i..][0..4].*;
        acc += va * vb;
    }

    var sum = @reduce(.Add, acc);

    while (i < len) : (i += 1) {
        sum += a_ptr[i] * b_ptr[i];
    }

    return sum;
}

/// Clip values in-place to [lo, hi]
pub export fn clip_f64(ptr: [*]f64, len: usize, lo: f64, hi: f64) void {
    for (0..len) |i| {
        const x = ptr[i];
        ptr[i] = @max(lo, @min(hi, x));
    }
}

/// Find index of maximum value
pub export fn argmax_f64(ptr: [*]const f64, len: usize) usize {
    if (len == 0) return 0;

    var max_idx: usize = 0;
    var max_val: f64 = ptr[0];

    for (1..len) |i| {
        if (ptr[i] > max_val) {
            max_val = ptr[i];
            max_idx = i;
        }
    }

    return max_idx;
}

// =============================================================================
// Multi-Threaded Variants
// =============================================================================

fn sumSqWorker(ptr: [*]const f64, start: usize, end: usize, out: *f64) void {
    out.* = sumSqSimd(ptr + start, end - start);
}

/// Multi-threaded sum of squares
/// n_threads = 0 means auto-detect CPU count
pub export fn sum_sq_f64_mt(ptr: [*]const f64, len: usize, n_threads_in: u32) f64 {
    var n_threads: u32 = n_threads_in;

    if (n_threads == 0) {
        n_threads = @intCast(std.Thread.getCpuCount() catch 1);
    }

    // Fall back to single-threaded for small arrays or single thread
    if (n_threads < 2 or len < 1_000_000) {
        return sumSqSimd(ptr, len);
    }

    if (n_threads > 64) n_threads = 64;

    var partial: [64]f64 = [_]f64{0.0} ** 64;
    var handles: [64]?std.Thread = [_]?std.Thread{null} ** 64;

    const chunk: usize = len / n_threads;

    // Spawn worker threads
    for (0..n_threads) |t| {
        const start = t * chunk;
        const end = if (t == n_threads - 1) len else start + chunk;

        handles[t] = std.Thread.spawn(.{}, sumSqWorker, .{ ptr, start, end, &partial[t] }) catch null;

        if (handles[t] == null) {
            // Fallback: run in current thread if spawn fails
            sumSqWorker(ptr, start, end, &partial[t]);
        }
    }

    // Join all threads
    for (0..n_threads) |t| {
        if (handles[t]) |h| h.join();
    }

    // Sum partial results
    var sum: f64 = 0.0;
    for (0..n_threads) |t| {
        sum += partial[t];
    }

    return sum;
}

// =============================================================================
// Tests
// =============================================================================

test "sum_sq_f64 basic" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const result = sum_sq_f64(&data, data.len);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), result, 1e-10);
}

test "dot_f64 basic" {
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f64{ 2.0, 3.0, 4.0, 5.0 };
    const result = dot_f64(&a, &b, a.len);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    try std.testing.expectApproxEqAbs(@as(f64, 40.0), result, 1e-10);
}

test "clip_f64 basic" {
    var data = [_]f64{ -1.0, 0.5, 1.5, 3.0 };
    clip_f64(&data, data.len, 0.0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), data[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), data[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), data[3], 1e-10);
}

test "argmax_f64 basic" {
    const data = [_]f64{ 1.0, 4.0, 2.0, 3.0 };
    const result = argmax_f64(&data, data.len);
    try std.testing.expectEqual(@as(usize, 1), result);
}
