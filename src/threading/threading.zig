const std = @import("std");
const simd = @import("../simd/simd.zig");
const activations = @import("../kernels/activations.zig");
const Vec4 = simd.Vec4;

// =============================================================================
// Threading Configuration
// =============================================================================

/// Maximum number of worker threads
pub const MAX_THREADS: u32 = 64;

/// Minimum array size for multi-threading (below this, single-threaded is faster)
pub const MT_THRESHOLD: usize = 1_000_000;

// =============================================================================
// Multi-Threaded Kernels
// =============================================================================

fn sumSqWorker(ptr: [*]const f64, start: usize, end: usize, out: *f64) void {
    out.* = simd.sumSqSimd(ptr + start, end - start);
}

fn softmaxMaxWorker(ptr: [*]const f64, start: usize, end: usize, out: *f64) void {
    var max_val: f64 = ptr[start];
    var i: usize = start + 1;
    while (i + 4 <= end) : (i += 4) {
        const v: Vec4 = ptr[i..][0..4].*;
        const chunk_max = @reduce(.Max, v);
        if (chunk_max > max_val) max_val = chunk_max;
    }
    while (i < end) : (i += 1) {
        if (ptr[i] > max_val) max_val = ptr[i];
    }
    out.* = max_val;
}

fn softmaxExpSumWorker(
    ptr: [*]f64,
    start: usize,
    end: usize,
    max_val: f64,
    out: *f64,
) void {
    var sum_vec: Vec4 = @splat(0.0);
    const max_splat: Vec4 = @splat(max_val);
    var i: usize = start;
    while (i + 4 <= end) : (i += 4) {
        const v: Vec4 = ptr[i..][0..4].*;
        const exp_v = @exp(v - max_splat);
        ptr[i..][0..4].* = exp_v;
        sum_vec += exp_v;
    }
    var sum = @reduce(.Add, sum_vec);
    while (i < end) : (i += 1) {
        ptr[i] = @exp(ptr[i] - max_val);
        sum += ptr[i];
    }
    out.* = sum;
}

fn softmaxScaleWorker(
    ptr: [*]f64,
    start: usize,
    end: usize,
    inv_sum: f64,
) void {
    const inv_sum_vec: Vec4 = @splat(inv_sum);
    var i: usize = start;
    while (i + 4 <= end) : (i += 4) {
        const v: Vec4 = ptr[i..][0..4].*;
        ptr[i..][0..4].* = v * inv_sum_vec;
    }
    while (i < end) : (i += 1) {
        ptr[i] *= inv_sum;
    }
}

/// Multi-threaded sum of squares
/// n_threads = 0 means auto-detect CPU count
pub fn sumSqMt(ptr: [*]const f64, len: usize, n_threads_in: u32) f64 {
    var n_threads: u32 = n_threads_in;

    if (n_threads == 0) {
        n_threads = @intCast(std.Thread.getCpuCount() catch 1);
    }

    // Fall back to single-threaded for small arrays or single thread
    if (n_threads < 2 or len < MT_THRESHOLD) {
        return simd.sumSqSimd(ptr, len);
    }

    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;

    var partial: [MAX_THREADS]f64 = [_]f64{0.0} ** MAX_THREADS;
    var handles: [MAX_THREADS]?std.Thread = [_]?std.Thread{null} ** MAX_THREADS;

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

/// Multi-threaded softmax (in-place)
/// n_threads = 0 means auto-detect CPU count
pub fn softmaxMt(ptr: [*]f64, len: usize, n_threads_in: u32) void {
    if (len == 0) return;

    var n_threads: u32 = n_threads_in;
    if (n_threads == 0) {
        n_threads = @intCast(std.Thread.getCpuCount() catch 1);
    }
    if (n_threads < 2 or len < MT_THRESHOLD) {
        activations.softmax(ptr, len);
        return;
    }
    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
    if (n_threads > len) n_threads = @intCast(len);

    var partial: [MAX_THREADS]f64 = [_]f64{0.0} ** MAX_THREADS;
    var handles: [MAX_THREADS]?std.Thread = [_]?std.Thread{null} ** MAX_THREADS;
    const chunk: usize = len / n_threads;

    // Pass 1: local maxima
    for (0..n_threads) |t| {
        const start = t * chunk;
        const end = if (t == n_threads - 1) len else start + chunk;
        handles[t] = std.Thread.spawn(.{}, softmaxMaxWorker, .{
            ptr,
            start,
            end,
            &partial[t],
        }) catch null;
        if (handles[t] == null) {
            softmaxMaxWorker(ptr, start, end, &partial[t]);
        }
    }
    for (0..n_threads) |t| {
        if (handles[t]) |h| h.join();
    }

    var max_val = partial[0];
    for (1..n_threads) |t| {
        if (partial[t] > max_val) max_val = partial[t];
    }

    // Pass 2: exp and local sums
    for (0..n_threads) |t| {
        const start = t * chunk;
        const end = if (t == n_threads - 1) len else start + chunk;
        handles[t] = std.Thread.spawn(.{}, softmaxExpSumWorker, .{
            ptr,
            start,
            end,
            max_val,
            &partial[t],
        }) catch null;
        if (handles[t] == null) {
            softmaxExpSumWorker(ptr, start, end, max_val, &partial[t]);
        }
    }
    for (0..n_threads) |t| {
        if (handles[t]) |h| h.join();
    }

    var sum: f64 = 0.0;
    for (0..n_threads) |t| {
        sum += partial[t];
    }

    // Pass 3: normalize
    const inv_sum = 1.0 / sum;
    for (0..n_threads) |t| {
        const start = t * chunk;
        const end = if (t == n_threads - 1) len else start + chunk;
        handles[t] = std.Thread.spawn(.{}, softmaxScaleWorker, .{
            ptr,
            start,
            end,
            inv_sum,
        }) catch null;
        if (handles[t] == null) {
            softmaxScaleWorker(ptr, start, end, inv_sum);
        }
    }
    for (0..n_threads) |t| {
        if (handles[t]) |h| h.join();
    }
}

// =============================================================================
// Tests
// =============================================================================

test "sumSqMt basic" {
    // Use a small array that will fall back to single-threaded
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const result = sumSqMt(&data, data.len, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), result, 1e-10);
}

test "sumSqMt with explicit thread count" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const result = sumSqMt(&data, data.len, 2);
    // 1+4+9+16+25+36+49+64 = 204
    try std.testing.expectApproxEqAbs(@as(f64, 204.0), result, 1e-10);
}

test "softmaxMt basic" {
    var data = [_]f64{ 1.0, 2.0, 3.0 };
    softmaxMt(&data, data.len, 2);
    const sum = data[0] + data[1] + data[2];
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-10);
    try std.testing.expect(data[0] < data[1]);
    try std.testing.expect(data[1] < data[2]);
}
