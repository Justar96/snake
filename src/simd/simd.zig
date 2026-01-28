
const std = @import("std");
const build_options = @import("build_options");

// =============================================================================
// SIMD Submodules
// =============================================================================

/// SIMD reduction helpers (sum, sumSq, dot)
pub const reduce = @import("reduce.zig");

// =============================================================================
// SIMD Type Aliases
// =============================================================================

/// Best-effort SIMD width for f64 kernels on this target
pub const VecF64Len: usize = std.simd.suggestVectorLength(f64) orelse 1;

/// Best-effort SIMD width for f32 kernels on this target
pub const VecF32Len: usize = std.simd.suggestVectorLength(f32) orelse 1;

/// Preferred f64 SIMD vector for kernels (width varies by target)
pub const VecF64 = @Vector(VecF64Len, f64);

/// Preferred f32 SIMD vector for kernels (width varies by target)
pub const VecF32 = @Vector(VecF32Len, f32);

/// 4-wide f64 SIMD vector - legacy fixed-width type
pub const Vec4 = @Vector(4, f64);

/// 4-wide f32 SIMD vector - legacy fixed-width type
pub const Vec4f32 = @Vector(4, f32);

// Optional SIMD width diagnostics for benchmarks/debugging.
pub const SIMD_DIAG: bool = build_options.simd_diag;

comptime {
    if (SIMD_DIAG) {
        @compileLog("snake SIMD width: f64 lanes=", VecF64Len, " f32 lanes=", VecF32Len);
        if (VecF64Len == 0 or VecF32Len == 0) {
            @compileError("SIMD lane count must be non-zero");
        }
    }
}

// =============================================================================
// SIMD Helper Functions
// =============================================================================

/// Sum of squares using SIMD: Σ(x²)
/// This is a core building block used by multiple kernels (sum_sq, normalize)
/// Delegates to reduce.sumSq for implementation.
pub const sumSqSimd = reduce.sumSq;
