# Argmax + Softmax Optimization Plan

## Scope
Target `argmax` and `softmax` to reach ≥0.8× NumPy single-thread performance
and set a foundation for ≥1.5× with multi-threaded variants.

## Profiling Summary (Static)

### Argmax (before)
- Scalar loop with a branch per element.
- No SIMD reduction; branch mispredicts dominate on random data.

### Softmax (before)
- Three full passes over memory: max, exp+sum, normalize.
- Exp dominates runtime; SIMD only on Vec4 and single accumulator.

## NumPy Baseline

NumPy's `argmax` runs in optimized C loops and typically leverages vectorized
reduce logic in its core C implementation. Softmax is commonly built from
NumPy/SciPy primitives (max-shift, exp, sum) that are vectorized in C.

## Implemented Optimizations

### Argmax
- SIMD vector compare with per-lane indices (`Vec4` + `@select`)
- Scalar tail for leftover elements
- Keeps first occurrence on ties by using `>`

Code changes:
- `src/kernels/reductions.zig`: vectorized argmax loop + lane reduction

### Softmax
- Small-length scalar fast path
- Unrolled SIMD loops (8 elements per iteration)
- Dual accumulators to reduce dependency chain in sum

Code changes:
- `src/kernels/activations.zig`: unrolled max pass + exp/sum + normalize
- `src/kernels/activations_test.zig`: tests moved to keep file size small
- `src/kernels/kernels.zig`: include activations tests

## Expected Improvements

| Kernel  | Expected Impact | Notes |
|---------|-----------------|-------|
| argmax  | 2–4× over scalar | Less branching, better ILP |
| softmax | 1.2–1.8× on large arrays | Exp still dominates |

Actual gains depend on CPU vector width, exp implementation, and cache.

## Trade-offs
- More complex SIMD logic and index handling.
- Unrolling increases code size slightly.
- No multi-threaded variants yet.

## Next-Step Proposals

1. **Multi-threaded softmax**: parallel max + sum reductions, then normalize.
2. **f32 variants**: match ML workloads and improve bandwidth.
3. **Wider SIMD dispatch**: AVX2/AVX-512 when available.
4. **Fast exp approximation** (optional): for throughput-focused workloads.

## softmax_mt Sketch

Goal: keep the same numerical stability while scaling on large arrays.

1. **Partition** input into N chunks (N = min(cpu_count, max_threads)).
2. **Pass 1 (max)**: each worker computes local max for its chunk.
3. **Reduce max**: main thread computes global max from local maxima.
4. **Pass 2 (exp + sum)**: each worker computes exp(x - max) in-place and
   accumulates a local sum.
5. **Reduce sum**: main thread computes global sum.
6. **Pass 3 (normalize)**: each worker scales its chunk by 1 / sum.

Notes:
- All passes are read/write in-place and are bandwidth-bound.
- Parallel overhead is only worth it above a size threshold (tune like `sum_sq_mt`).
- Be explicit about tie-breaking and NaN handling when defining correctness.
- For MKL-backed NumPy, parity may require multi-threading on large arrays.

Prototype status:
- Implemented in `src/threading/threading.zig` as `softmaxMt`
- C ABI export: `softmax_f64_mt` in `src/snake.zig`
- Python wrapper: `softmax_mt` in `python/snake/_core.py`

## Validation

- Zig tests: `zig build test`
- Python parity tests: `uv run pytest tests/`
- Benchmarks: `python bench/bench.py` (Phase 1 kernels)
