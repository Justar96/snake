const std = @import("std");
const simd = @import("../simd/simd.zig");
const Vec = simd.VecF64;
const VecLen = simd.VecF64Len;
/// ReLU activation in-place: x[i] = max(0, x[i])
pub fn relu(ptr: [*]f64, len: usize) void {
    const zero: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + VecLen <= len) : (i += VecLen) {
        const v: Vec = ptr[i..][0..VecLen].*;
        ptr[i..][0..VecLen].* = @max(zero, v);
    }
    while (i < len) : (i += 1) {
        ptr[i] = @max(0.0, ptr[i]);
    }
}
/// GELU activation in-place using tanh approximation.
pub fn gelu(ptr: [*]f64, len: usize) void {
    const sqrt_2_pi: Vec = @splat(0.7978845608028654); // sqrt(2/π)
    const coeff: Vec = @splat(0.044715);
    const half: Vec = @splat(0.5);
    const one: Vec = @splat(1.0);
    const two: Vec = @splat(2.0);
    var i: usize = 0;
    while (i + VecLen <= len) : (i += VecLen) {
        const x: Vec = ptr[i..][0..VecLen].*;
        const x2 = x * x;
        const x3 = x2 * x;
        const inner = sqrt_2_pi * (x + coeff * x3);
        const exp_2t = @exp(two * inner);
        const tanh_val = (exp_2t - one) / (exp_2t + one);
        ptr[i..][0..VecLen].* = half * x * (one + tanh_val);
    }
    const sqrt_2_pi_scalar: f64 = 0.7978845608028654;
    const coeff_scalar: f64 = 0.044715;
    while (i < len) : (i += 1) {
        const x = ptr[i];
        const x3 = x * x * x;
        const inner = sqrt_2_pi_scalar * (x + coeff_scalar * x3);
        ptr[i] = 0.5 * x * (1.0 + std.math.tanh(inner));
    }
}
/// Softmax activation with 3-pass approach and 8-wide unrolling.
pub fn softmax(ptr: [*]f64, len: usize) void {
    if (len == 0) return;
    if (len < VecLen) {
        var max_val: f64 = ptr[0];
        for (1..len) |idx| {
            if (ptr[idx] > max_val) max_val = ptr[idx];
        }
        var sum: f64 = 0.0;
        for (0..len) |idx| {
            ptr[idx] = @exp(ptr[idx] - max_val);
            sum += ptr[idx];
        }
        const inv_sum = 1.0 / sum;
        for (0..len) |idx| {
            ptr[idx] *= inv_sum;
        }
        return;
    }
    const step = VecLen * 2;

    // Single-vector path for sizes VecLen..(2*VecLen-1).
    if (len < step) {
        const v0: Vec = ptr[0..VecLen].*;
        var max_val = @reduce(.Max, v0);
        var i: usize = VecLen;
        while (i < len) : (i += 1) {
            if (ptr[i] > max_val) max_val = ptr[i];
        }
        const max_splat: Vec = @splat(max_val);
        const exp0 = @exp(v0 - max_splat);
        ptr[0..VecLen].* = exp0;
        var sum = @reduce(.Add, exp0);
        i = VecLen;
        while (i < len) : (i += 1) {
            ptr[i] = @exp(ptr[i] - max_val);
            sum += ptr[i];
        }
        const inv_sum = 1.0 / sum;
        const inv_sum_vec: Vec = @splat(inv_sum);
        ptr[0..VecLen].* = ptr[0..VecLen].* * inv_sum_vec;
        i = VecLen;
        while (i < len) : (i += 1) {
            ptr[i] *= inv_sum;
        }
        return;
    }

    var i: usize = 0;
    var max_vec0: Vec = ptr[0..VecLen].*;
    var max_vec1: Vec = ptr[VecLen..step].*;
    i = step;
    while (i + step <= len) : (i += step) {
        const v0: Vec = ptr[i..][0..VecLen].*;
        const v1: Vec = ptr[i + VecLen ..][0..VecLen].*;
        max_vec0 = @max(max_vec0, v0);
        max_vec1 = @max(max_vec1, v1);
    }
    if (i + VecLen <= len) {
        const v: Vec = ptr[i..][0..VecLen].*;
        max_vec0 = @max(max_vec0, v);
        i += VecLen;
    }
    var max_val = @reduce(.Max, @max(max_vec0, max_vec1));
    while (i < len) : (i += 1) {
        if (ptr[i] > max_val) max_val = ptr[i];
    }

    i = 0;
    var sum_vec0: Vec = @splat(0.0);
    var sum_vec1: Vec = @splat(0.0);
    const max_splat: Vec = @splat(max_val);
    while (i + step <= len) : (i += step) {
        const v0: Vec = ptr[i..][0..VecLen].*;
        const v1: Vec = ptr[i + VecLen ..][0..VecLen].*;
        const exp0 = @exp(v0 - max_splat);
        const exp1 = @exp(v1 - max_splat);
        ptr[i..][0..VecLen].* = exp0;
        ptr[i + VecLen ..][0..VecLen].* = exp1;
        sum_vec0 += exp0;
        sum_vec1 += exp1;
    }
    if (i + VecLen <= len) {
        const v: Vec = ptr[i..][0..VecLen].*;
        const exp_v = @exp(v - max_splat);
        ptr[i..][0..VecLen].* = exp_v;
        sum_vec0 += exp_v;
        i += VecLen;
    }
    var sum = @reduce(.Add, sum_vec0 + sum_vec1);
    while (i < len) : (i += 1) {
        ptr[i] = @exp(ptr[i] - max_val);
        sum += ptr[i];
    }

    i = 0;
    const inv_sum = 1.0 / sum;
    const inv_sum_vec: Vec = @splat(inv_sum);
    while (i + step <= len) : (i += step) {
        const v0: Vec = ptr[i..][0..VecLen].*;
        const v1: Vec = ptr[i + VecLen ..][0..VecLen].*;
        ptr[i..][0..VecLen].* = v0 * inv_sum_vec;
        ptr[i + VecLen ..][0..VecLen].* = v1 * inv_sum_vec;
    }
    if (i + VecLen <= len) {
        const v: Vec = ptr[i..][0..VecLen].*;
        ptr[i..][0..VecLen].* = v * inv_sum_vec;
        i += VecLen;
    }
    while (i < len) : (i += 1) {
        ptr[i] *= inv_sum;
    }
}
