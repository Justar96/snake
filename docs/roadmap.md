# snake Roadmap

## Phase 0: POC (Current) ✅

**Goal**: Prove the concept with minimal viable kernels.

### Deliverables

- [x] Project structure with `build.zig`
- [x] Core kernels: `sum_sq`, `dot`, `clip`, `argmax`
- [x] Multi-threaded `sum_sq_mt`
- [x] Python ctypes bindings
- [x] Benchmark harness

### Success Criteria

- `sum_sq` ≥ 20× faster than Python loop
- `sum_sq` within 1.2× of NumPy (single-threaded)
- `sum_sq_mt` beats NumPy on multi-core systems

---

## Phase 1: Core Kernel Library

**Goal**: Build a useful set of vectorized operations.

### New Kernels

| Kernel            | Description                 | Priority |
| ----------------- | --------------------------- | -------- |
| `normalize_f64`   | L2 normalize in-place       | High     |
| `scale_f64`       | Multiply by scalar in-place | High     |
| `saxpy_f64`       | a = a + scalar \* b         | High     |
| `softmax_f64`     | Softmax activation          | Medium   |
| `relu_f64`        | ReLU activation in-place    | Medium   |
| `gelu_f64`        | GELU activation             | Medium   |
| `rolling_sum_f64` | Windowed sum                | Medium   |
| `cumsum_f64`      | Cumulative sum              | Medium   |
| `variance_f64`    | Welford's online variance   | Low      |
| `histogram_f64`   | Binned histogram            | Low      |

### Multi-Threaded Variants

- All kernels that operate on large buffers should have `_mt` variants
- Threshold tuning: determine optimal crossover point per kernel

---

## Phase 2: Data Type Support

**Goal**: Support common numeric types beyond float64.

### Types

- [ ] `f32` (float32) — Important for ML workloads
- [ ] `i64` (int64) — Important for indexing
- [ ] `i32` (int32)
- [ ] `u8` (uint8) — Image processing

### Implementation

- Create generic Zig functions with comptime type parameter
- Python side: detect dtype and dispatch to appropriate function

---

## Phase 3: Distribution

**Goal**: Easy installation via pip.

### Tasks

- [ ] Build system for cross-platform compilation
- [ ] Pre-built wheels for:
  - Linux x86_64
  - Linux aarch64
  - macOS x86_64
  - macOS arm64 (Apple Silicon)
  - Windows x86_64
- [ ] GitHub Actions CI/CD
- [ ] PyPI publishing

### Build Approach Options

1. **scikit-build-core** — Cleanest integration with pyproject.toml
2. **cibuildwheel** — Multi-platform wheel generation
3. **maturin** — If we ever want Rust interop too

---

## Phase 4: Advanced Features

### SIMD Dispatch

- [ ] Runtime CPU feature detection (AVX, AVX2, AVX-512, NEON)
- [ ] Select optimal implementation at load time

### Streaming/Chunked Processing

- [ ] Iterator-based API for datasets larger than RAM
- [ ] Memory-mapped file support

### Python 3.14 Free-Threading

- [ ] Declare `Py_mod_gil = Py_MOD_GIL_NOT_USED`
- [ ] Test with free-threaded Python builds
- [ ] Benchmark true parallel Python threads calling Zig

---

## Phase 5: Ecosystem Integration

### NumPy Universal Functions (ufuncs)

- [ ] Expose kernels as NumPy ufuncs
- [ ] Support broadcasting

### Pandas Extension

- [ ] Custom Pandas ExtensionArray backed by Zig
- [ ] Optimized aggregations

### Polars Plugin

- [ ] Native Polars expression plugin
- [ ] Zero-copy Arrow interop

---

## Non-Goals (Out of Scope)

- **Full BLAS replacement** — Use OpenBLAS/MKL for matrix ops
- **GPU compute** — Focus on CPU SIMD (may revisit later)
- **Complex numbers** — Keep it simple for v1
- **Sparse arrays** — Different problem domain

---

## Timeline (Rough Estimate)

| Phase   | Duration  | Target     |
| ------- | --------- | ---------- |
| Phase 0 | 1 week    | ✅ Done    |
| Phase 1 | 2-3 weeks | Q1 2026    |
| Phase 2 | 1-2 weeks | Q1 2026    |
| Phase 3 | 2-3 weeks | Q2 2026    |
| Phase 4 | Ongoing   | Q2-Q3 2026 |
| Phase 5 | Ongoing   | Q3+ 2026   |
