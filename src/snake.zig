// =============================================================================
// Snake - SIMD-Vectorized Numeric Operations
// =============================================================================
//
// Root module that provides the C ABI interface and internal modules.
//
// Module Structure:
//   src/
//   ├── snake.zig           # This file - root module with C ABI exports
//   ├── simd/
//   │   └── simd.zig        # SIMD type aliases and helpers
//   ├── kernels/
//   │   ├── kernels.zig     # Re-exports all kernel modules
//   │   ├── reductions.zig  # sum_sq, dot, variance, argmax
//   │   ├── transforms.zig  # clip, scale, normalize, saxpy
//   │   ├── activations.zig # relu, gelu, softmax
//   │   ├── prefix.zig      # cumsum, rolling_sum
//   │   └── histogram.zig   # histogram
//   └── threading/
//       └── threading.zig   # Multi-threaded variants
//
// =============================================================================

// Internal modules (for direct Zig usage)
pub const simd = @import("simd/simd.zig");
pub const kernels = @import("kernels/kernels.zig");
pub const threading = @import("threading/threading.zig");

// =============================================================================
// C ABI Exports - Reduction Kernels
// =============================================================================

/// Sum of squares: Σ(x²)
pub export fn sum_sq_f64(ptr: [*]const f64, len: usize) f64 {
    return kernels.reductions.sumSq(ptr, len);
}

/// Dot product: Σ(a · b)
pub export fn dot_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, len: usize) f64 {
    return kernels.reductions.dot(a_ptr, b_ptr, len);
}

/// Find index of maximum value
pub export fn argmax_f64(ptr: [*]const f64, len: usize) usize {
    return kernels.reductions.argmax(ptr, len);
}

/// Variance using Welford's online algorithm
pub export fn variance_f64(ptr: [*]const f64, len: usize) f64 {
    return kernels.reductions.variance(ptr, len);
}

// =============================================================================
// C ABI Exports - Transform Kernels
// =============================================================================

/// Clip values in-place to [lo, hi]
pub export fn clip_f64(ptr: [*]f64, len: usize, lo: f64, hi: f64) void {
    kernels.transforms.clip(ptr, len, lo, hi);
}

/// L2 normalize in-place
pub export fn normalize_f64(ptr: [*]f64, len: usize) void {
    kernels.transforms.normalize(ptr, len);
}

/// Multiply array by scalar in-place
pub export fn scale_f64(ptr: [*]f64, len: usize, scalar: f64) void {
    kernels.transforms.scale(ptr, len, scalar);
}

/// SAXPY: a[i] += scalar * b[i]
pub export fn saxpy_f64(a_ptr: [*]f64, b_ptr: [*]const f64, len: usize, scalar: f64) void {
    kernels.transforms.saxpy(a_ptr, b_ptr, len, scalar);
}

// =============================================================================
// C ABI Exports - Activation Kernels
// =============================================================================

/// ReLU activation in-place
pub export fn relu_f64(ptr: [*]f64, len: usize) void {
    kernels.activations.relu(ptr, len);
}

/// GELU activation in-place using tanh approximation
pub export fn gelu_f64(ptr: [*]f64, len: usize) void {
    kernels.activations.gelu(ptr, len);
}

/// Softmax activation in-place
pub export fn softmax_f64(ptr: [*]f64, len: usize) void {
    kernels.activations.softmax(ptr, len);
}

// =============================================================================
// C ABI Exports - Prefix/Rolling Kernels
// =============================================================================

/// Cumulative sum (prefix sum) in-place
pub export fn cumsum_f64(ptr: [*]f64, len: usize) void {
    kernels.prefix.cumsum(ptr, len);
}

/// Rolling sum with fixed window size
pub export fn rolling_sum_f64(
    in_ptr: [*]const f64,
    out_ptr: [*]f64,
    len: usize,
    window: usize,
) void {
    kernels.prefix.rollingSum(in_ptr, out_ptr, len, window);
}

// =============================================================================
// C ABI Exports - Histogram Kernel
// =============================================================================

/// Binned histogram
pub export fn histogram_f64(
    data_ptr: [*]const f64,
    len: usize,
    bins_ptr: [*]f64,
    n_bins: usize,
    min_val: f64,
    max_val: f64,
) void {
    kernels.histogram.histogram(data_ptr, len, bins_ptr, n_bins, min_val, max_val);
}

// =============================================================================
// C ABI Exports - Multi-Threaded Kernels
// =============================================================================

/// Multi-threaded sum of squares
pub export fn sum_sq_f64_mt(ptr: [*]const f64, len: usize, n_threads: u32) f64 {
    return threading.sumSqMt(ptr, len, n_threads);
}

/// Multi-threaded softmax activation in-place
pub export fn softmax_f64_mt(ptr: [*]f64, len: usize, n_threads: u32) void {
    threading.softmaxMt(ptr, len, n_threads);
}

// =============================================================================
// Test imports - ensures all module tests are run
// =============================================================================

test {
    _ = simd;
    _ = kernels;
    _ = threading;
}
