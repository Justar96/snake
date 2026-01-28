# snake Architecture

## Overview

zigops provides high-performance numeric kernels written in Zig, callable from Python via ctypes. The design prioritizes:

1. **Zero-copy buffer access** — NumPy arrays are accessed directly via pointer
2. **Explicit SIMD** — Uses Zig's `@Vector` for guaranteed vectorization
3. **GIL-free execution** — ctypes calls release the GIL during foreign function execution
4. **Native threading** — Multi-threaded variants use Zig's threading primitives

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Application                       │
├─────────────────────────────────────────────────────────────┤
│                      numpy.ndarray                           │
│                 (contiguous float64 buffer)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │ ctypes.POINTER
                          │ (releases GIL)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   zigops/_core.py                            │
│  • Locates shared library                                   │
│  • Defines function signatures                              │
│  • Converts arrays to (ptr, len) tuples                     │
└─────────────────────────┬───────────────────────────────────┘
                          │ C ABI call
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   libzigops.so                               │
│  • sum_sq_f64(ptr, len) -> f64                              │
│  • sum_sq_f64_mt(ptr, len, n_threads) -> f64                │
│  • dot_f64(a, b, len) -> f64                                │
│  • clip_f64(ptr, len, lo, hi) -> void                       │
│  • argmax_f64(ptr, len) -> usize                            │
├─────────────────────────────────────────────────────────────┤
│  Internal:                                                   │
│  • @Vector(4, f64) SIMD primitives                          │
│  • std.Thread for parallel execution                        │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Single-Threaded Kernel (e.g., `sum_sq`)

```
numpy.ndarray
    │
    ▼ np.ascontiguousarray(dtype=float64)
    │
    ▼ .ctypes.data_as(POINTER(c_double))
    │
┌───▼───────────────────────────────────────┐
│  sum_sq_f64(ptr, len)                     │
│                                           │
│  1. SIMD loop: process 4 elements/iter    │
│     acc += v * v  (vectorized)            │
│                                           │
│  2. Scalar tail: remaining 0-3 elements   │
│                                           │
│  3. @reduce(.Add, acc) → scalar           │
└───────────────────────────────────────────┘
    │
    ▼ float (Python)
```

### Multi-Threaded Kernel (e.g., `sum_sq_mt`)

```
┌───────────────────────────────────────────────────────────────┐
│  sum_sq_f64_mt(ptr, len, n_threads)                           │
│                                                               │
│  1. Determine thread count (or auto-detect CPU count)         │
│  2. Check threshold: if len < 1M, fall back to single-thread  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │        Thread Pool (up to 64 workers)                   │  │
│  │                                                         │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐   │  │
│  │  │Worker 0 │ │Worker 1 │ │Worker 2 │ ... │Worker N │   │  │
│  │  │chunk 0  │ │chunk 1  │ │chunk 2  │     │chunk N  │   │  │
│  │  │→partial0│ │→partial1│ │→partial2│     │→partialN│   │  │
│  │  └─────────┘ └─────────┘ └─────────┘     └─────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  3. Join all threads                                          │
│  4. Sum partial results                                       │
└───────────────────────────────────────────────────────────────┘
```

## SIMD Strategy

Zig's `@Vector` provides portable SIMD without architecture-specific intrinsics:

```zig
const Vec4 = @Vector(4, f64);

// Loads 4 contiguous f64 values
const v: Vec4 = ptr[i..][0..4].*;

// SIMD multiply + accumulate (compiles to SIMD instructions)
acc += v * v;

// Horizontal reduction
const sum = @reduce(.Add, acc);
```

This compiles to:

- **x86_64**: AVX/AVX2 instructions (`vfmadd`, `vaddpd`)
- **ARM64**: NEON instructions

## Memory Safety

1. **No allocations** — All kernels operate on caller-provided buffers
2. **Bounds checked** — Loop indices never exceed `len`
3. **No aliasing** — Input/output pointers are distinct (enforced by API design)

## Future Extensions

- [ ] `f32` variants for half-precision workloads
- [ ] AVX-512 detection and wider vectors
- [ ] GPU offload via SYCL or CUDA interop
- [ ] Streaming/chunked processing for datasets larger than RAM
