const std = @import("std");

// =============================================================================
// Prefix/Rolling Operations
// =============================================================================

/// Cumulative sum (prefix sum): out[i] = Σ(x[0..i+1])
pub fn cumsum(ptr: [*]f64, len: usize) void {
    if (len == 0) return;

    var acc: f64 = ptr[0];
    for (1..len) |i| {
        acc += ptr[i];
        ptr[i] = acc;
    }
}

/// Rolling sum with fixed window size
/// out[i] = sum(in[max(0, i-window+1)..i+1])
pub fn rollingSum(
    in_ptr: [*]const f64,
    out_ptr: [*]f64,
    len: usize,
    window: usize,
) void {
    if (len == 0 or window == 0) return;

    // Compute first window using running sum
    var sum: f64 = 0.0;
    const actual_window = if (window > len) len else window;

    for (0..len) |i| {
        sum += in_ptr[i];
        if (i >= actual_window) {
            sum -= in_ptr[i - actual_window];
        }
        out_ptr[i] = sum;
    }
}

// =============================================================================
// Tests
// =============================================================================

test "cumsum basic" {
    var data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    cumsum(&data, data.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), data[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), data[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), data[3], 1e-10);
}

test "rollingSum basic" {
    const input = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var output: [5]f64 = undefined;
    rollingSum(&input, &output, input.len, 3);
    // Window 3: [1], [1,2], [1,2,3], [2,3,4], [3,4,5]
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), output[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), output[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), output[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), output[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), output[4], 1e-10);
}

test "cumsum empty" {
    var data = [_]f64{};
    cumsum(&data, 0);
    // Should not crash
}

test "rollingSum window larger than array" {
    const input = [_]f64{ 1.0, 2.0, 3.0 };
    var output: [3]f64 = undefined;
    rollingSum(&input, &output, input.len, 10);
    // Window > len: same as cumsum
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), output[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), output[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), output[2], 1e-10);
}
