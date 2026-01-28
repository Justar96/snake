// =============================================================================
// Kernels Module
// Re-exports all kernel implementations
// =============================================================================

pub const reductions = @import("reductions.zig");
pub const transforms = @import("transforms.zig");
pub const activations = @import("activations.zig");
pub const prefix = @import("prefix.zig");
pub const histogram = @import("histogram.zig");
pub const activations_test = @import("activations_test.zig");

// =============================================================================
// Test imports - ensures all kernel tests are run
// =============================================================================

test {
    _ = reductions;
    _ = transforms;
    _ = activations;
    _ = activations_test;
    _ = prefix;
    _ = histogram;
}
