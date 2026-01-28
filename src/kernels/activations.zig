const std = @import("std");
const simd = @import("../simd/simd.zig");
const Vec4 = simd.Vec4;
/// ReLU activation in-place: x[i] = max(0, x[i])
pub fn relu(ptr: [*]f64, len: usize) void {
    const zero: Vec4 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = ptr[i..][0..4].*;
        ptr[i..][0..4].* = @max(zero, v);
    }
    while (i < len) : (i += 1) {
        ptr[i] = @max(0.0, ptr[i]);
    }
}
/// GELU activation in-place using tanh approximation.
pub fn gelu(ptr: [*]f64, len: usize) void {
    const sqrt_2_pi: Vec4 = @splat(0.7978845608028654); // sqrt(2/π)
    const coeff: Vec4 = @splat(0.044715);
    const half: Vec4 = @splat(0.5);
    const one: Vec4 = @splat(1.0);
    const two: Vec4 = @splat(2.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const x: Vec4 = ptr[i..][0..4].*;
        const x2 = x * x;
        const x3 = x2 * x;
        const inner = sqrt_2_pi * (x + coeff * x3);
        const exp_2t = @exp(two * inner);
        const tanh_val = (exp_2t - one) / (exp_2t + one);
        ptr[i..][0..4].* = half * x * (one + tanh_val);
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
    if (len < 4) {
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
    // 4-wide SIMD path for sizes 4..7.
    if (len < 8) {
        const v0: Vec4 = ptr[0..4].*;
        var max_val = @reduce(.Max, v0);
        var i: usize = 4;
        while (i < len) : (i += 1) {
            if (ptr[i] > max_val) max_val = ptr[i];
        }
        const max_splat: Vec4 = @splat(max_val);
        const exp0 = @exp(v0 - max_splat);
        ptr[0..4].* = exp0;
        var sum = @reduce(.Add, exp0);
        i = 4;
        while (i < len) : (i += 1) {
            ptr[i] = @exp(ptr[i] - max_val);
            sum += ptr[i];
        }
        const inv_sum = 1.0 / sum;
        const inv_sum_vec: Vec4 = @splat(inv_sum);
        ptr[0..4].* = ptr[0..4].* * inv_sum_vec;
        i = 4;
        while (i < len) : (i += 1) {
            ptr[i] *= inv_sum;
        }
        return;
    }
    var i: usize = 0;
    var max_vec0: Vec4 = ptr[0..4].*;
    var max_vec1: Vec4 = ptr[4..8].*;
    i = 8;
    while (i + 8 <= len) : (i += 8) {
        const v0: Vec4 = ptr[i..][0..4].*;
        const v1: Vec4 = ptr[i + 4 ..][0..4].*;
        max_vec0 = @max(max_vec0, v0);
        max_vec1 = @max(max_vec1, v1);
    }
    if (i + 4 <= len) {
        const v: Vec4 = ptr[i..][0..4].*;
        max_vec0 = @max(max_vec0, v);
        i += 4;
    }
    var max_val = @reduce(.Max, @max(max_vec0, max_vec1));
    while (i < len) : (i += 1) {
        if (ptr[i] > max_val) max_val = ptr[i];
    }
    i = 0;
    var sum_vec0: Vec4 = @splat(0.0);
    var sum_vec1: Vec4 = @splat(0.0);
    const max_splat: Vec4 = @splat(max_val);
    while (i + 8 <= len) : (i += 8) {
        const v0: Vec4 = ptr[i..][0..4].*;
        const v1: Vec4 = ptr[i + 4 ..][0..4].*;
        const exp0 = @exp(v0 - max_splat);
        const exp1 = @exp(v1 - max_splat);
        ptr[i..][0..4].* = exp0;
        ptr[i + 4 ..][0..4].* = exp1;
        sum_vec0 += exp0;
        sum_vec1 += exp1;
    }
    if (i + 4 <= len) {
        const v: Vec4 = ptr[i..][0..4].*;
        const exp_v = @exp(v - max_splat);
        ptr[i..][0..4].* = exp_v;
        sum_vec0 += exp_v;
        i += 4;
    }
    var sum = @reduce(.Add, sum_vec0 + sum_vec1);
    while (i < len) : (i += 1) {
        ptr[i] = @exp(ptr[i] - max_val);
        sum += ptr[i];
    }
    i = 0;
    const inv_sum = 1.0 / sum;
    const inv_sum_vec: Vec4 = @splat(inv_sum);
    while (i + 8 <= len) : (i += 8) {
        const v0: Vec4 = ptr[i..][0..4].*;
        const v1: Vec4 = ptr[i + 4 ..][0..4].*;
        ptr[i..][0..4].* = v0 * inv_sum_vec;
        ptr[i + 4 ..][0..4].* = v1 * inv_sum_vec;
    }
    if (i + 4 <= len) {
        const v: Vec4 = ptr[i..][0..4].*;
        ptr[i..][0..4].* = v * inv_sum_vec;
        i += 4;
    }
    while (i < len) : (i += 1) {
        ptr[i] *= inv_sum;
    }
}
