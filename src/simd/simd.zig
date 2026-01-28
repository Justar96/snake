
// =============================================================================
// SIMD Submodules
// =============================================================================

/// SIMD reduction helpers (sum, sumSq, dot)
pub const reduce = @import("reduce.zig");

// =============================================================================
// SIMD Type Aliases
// =============================================================================

/// 4-wide f64 SIMD vector - primary vector type for double-precision kernels
pub const Vec4 = @Vector(4, f64);

/// 4-wide f32 SIMD vector - for future single-precision kernels
pub const Vec4f32 = @Vector(4, f32);

// =============================================================================
// SIMD Helper Functions
// =============================================================================

/// Sum of squares using SIMD: Σ(x²)
/// This is a core building block used by multiple kernels (sum_sq, normalize)
/// Delegates to reduce.sumSq for implementation.
pub const sumSqSimd = reduce.sumSq;

