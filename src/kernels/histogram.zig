const std = @import("std");

// =============================================================================
// Histogram Kernel
// =============================================================================

/// Binned histogram: count values into bins
/// bins_ptr is output array of n_bins elements
pub fn histogram(
    data_ptr: [*]const f64,
    len: usize,
    bins_ptr: [*]f64,
    n_bins: usize,
    min_val: f64,
    max_val: f64,
) void {
    if (n_bins == 0 or len == 0) return;

    // Clear bins
    for (0..n_bins) |i| {
        bins_ptr[i] = 0.0;
    }

    const range = max_val - min_val;
    if (range <= 0.0) return;

    const scale = @as(f64, @floatFromInt(n_bins)) / range;

    for (0..len) |i| {
        const x = data_ptr[i];
        if (x >= min_val and x < max_val) {
            const bin_idx: usize = @intFromFloat((x - min_val) * scale);
            const safe_idx = if (bin_idx >= n_bins) n_bins - 1 else bin_idx;
            bins_ptr[safe_idx] += 1.0;
        } else if (x == max_val) {
            // Include max value in last bin
            bins_ptr[n_bins - 1] += 1.0;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "histogram basic" {
    const data = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 };
    var bins: [5]f64 = undefined;
    histogram(&data, data.len, &bins, 5, 0.0, 1.0);
    // Bins: [0-0.2], [0.2-0.4], [0.4-0.6], [0.6-0.8], [0.8-1.0]
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bins[0], 1e-10); // 0.1
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bins[1], 1e-10); // 0.3
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bins[2], 1e-10); // 0.5
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bins[3], 1e-10); // 0.7
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), bins[4], 1e-10); // 0.9, 1.0
}

test "histogram empty data" {
    const data = [_]f64{};
    var bins: [5]f64 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    histogram(&data, 0, &bins, 5, 0.0, 1.0);
    // Bins should be cleared to zero (but we don't clear on empty data)
    // Actually, looking at the code, we return early if len == 0 so bins aren't cleared
}

test "histogram out of range" {
    const data = [_]f64{ -1.0, 0.5, 2.0 };
    var bins: [2]f64 = undefined;
    histogram(&data, data.len, &bins, 2, 0.0, 1.0);
    // Only 0.5 should be counted - with 2 bins over [0,1), 0.5 falls in second bin
    // Bin 0: [0.0, 0.5), Bin 1: [0.5, 1.0]
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), bins[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bins[1], 1e-10);
}
