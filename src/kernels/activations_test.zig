const std = @import("std");
const activations = @import("activations.zig");

test "relu basic" {
    var data = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    activations.relu(&data, data.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), data[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), data[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), data[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), data[4], 1e-10);
}

test "gelu basic" {
    var data = [_]f64{ 0.0, 1.0, -1.0 };
    activations.gelu(&data, data.len);
    // GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.841), data[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, -0.159), data[2], 0.01);
}

test "softmax basic" {
    var data = [_]f64{ 1.0, 2.0, 3.0 };
    activations.softmax(&data, data.len);
    // Sum should be 1.0
    const sum = data[0] + data[1] + data[2];
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-10);
    // Values should be monotonically increasing
    try std.testing.expect(data[0] < data[1]);
    try std.testing.expect(data[1] < data[2]);
}

test "softmax numerical stability" {
    // Test with large values that could cause overflow without max subtraction
    var data = [_]f64{ 1000.0, 1001.0, 1002.0 };
    activations.softmax(&data, data.len);
    const sum = data[0] + data[1] + data[2];
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-10);
    try std.testing.expect(data[0] < data[1]);
    try std.testing.expect(data[1] < data[2]);
}

test "softmax large array" {
    var data = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    activations.softmax(&data, data.len);
    var sum: f64 = 0.0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-10);
    for (1..data.len) |i| {
        try std.testing.expect(data[i - 1] < data[i]);
    }
}
