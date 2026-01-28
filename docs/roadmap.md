# snake Roadmap (2026)

This roadmap aligns Zig kernel development, Python bindings, and benchmark
coverage. Each phase includes explicit tests and benchmarks to keep progress
measurable and regression-safe.

## Guiding Principles

- Benchmark what runs on CPU in real Python inference stacks.
- Maintain parity checks against NumPy/PyTorch baselines where applicable.
- Keep kernels small, composable, and zero-copy (ctypes pointer + length).
- Treat benchmarks as first-class: every kernel ships with a benchmark target.

---

## Phase 0: Foundations (Done)

### Zig
- Core SIMD kernels: `sum_sq`, `dot`, `clip`, `argmax`
- Multi-threaded `sum_sq_mt`
- C ABI exports

### Python
- ctypes bindings in `python/snake/_core.py`
- Zero-copy NumPy access helpers

### Benchmarks
- `bench/bench.py` (sum_sq, dot, argmax)
- `docs/benchmarks.md` methodology

### Tests
- Zig unit tests for core kernels
- Python pytest coverage for bindings + dtype conversion

---

## Phase 1: Core Numeric Kernel Library (Done)

### Zig
- Vector math: normalize, scale, saxpy
- Activations: relu, gelu, softmax (all SIMD-optimized)
- Prefix/rolling ops: cumsum, rolling_sum
- Stats: variance, histogram (SIMD-optimized)

### Python
- Bindings for each kernel
- Input validation + dtype conversion helpers

### Benchmarks
- `bench/bench.py` includes all Phase 1 kernels
- Latest results (10M elements, vs NumPy):
  - **sum_sq**: 1.31× faster (180× vs pure Python)
  - **sum_sq_mt**: 1.53× faster (multi-threaded)
  - **dot**: 1.26× faster
  - **normalize**: 4.73× faster
  - **scale**: 1.54× faster
  - **saxpy**: 1.72× faster
  - **gelu**: 7.14× faster
  - **cumsum**: 2.08× faster
  - **rolling_sum**: 7.90× faster
- **variance**: 3.94× faster
- **histogram**: 6.63× faster
- **relu**: ~parity with NumPy
- **argmax**, **softmax**: SIMD/unrolled optimizations applied; re-benchmark for parity
  (see `docs/argmax_softmax_optimizations.md`)

### Tests
- Zig unit tests for each kernel (25 tests)
- Python parity tests vs NumPy (41 tests)
- Edge cases: empty, single element, NaN/Inf behavior

---

## Phase 1.5: Architecture Modernization (Done)

### Zig Module Structure
Refactored monolithic `snake.zig` (469 LOC) into modular structure:

```
src/
├── snake.zig              # Root module with C ABI exports (~130 LOC)
├── simd/
│   ├── simd.zig           # Vec4 type alias, sumSqSimd helper
│   └── reduce.zig         # Generic SIMD reduction helpers (sum, sumSq, dot)
├── kernels/
│   ├── kernels.zig        # Re-exports all kernel modules
│   ├── reductions.zig     # sum_sq, dot, argmax, variance
│   ├── transforms.zig     # clip, normalize, scale, saxpy
│   ├── activations.zig    # relu, gelu, softmax
│   ├── prefix.zig         # cumsum, rolling_sum
│   └── histogram.zig      # histogram
└── threading/
    └── threading.zig      # Multi-threaded sum_sq_mt
```

### Improvements
- All kernel files under ~150 LOC each
- Generic SIMD reduction helpers for code reuse
- Clean separation: internal Zig API vs C ABI exports
- Full SIMD vectorization for all applicable kernels

### Tests
- All 25 Zig tests pass across modules
- All 41 Python tests pass with refactored library

---

## Phase 2: LLM Microkernel Baselines (Done)

### Python
- `sample_token` baseline (top-k/top-p + repetition penalty)
- Tokenizer baseline: regex pretokenization + demo BPE loop
- RAG cosine scoring: exact + candidate list
- KV cache ring buffer updates

### Fixtures
- `bench/fixtures/bpe_demo.json` for demo BPE merges
- `bench/fixtures/tokenizer_vocab.json` + `tokenizer_merges.txt` for Rust BPE
- `bench/fixtures/tokenizer_corpus.txt` for deterministic fixture generation

### Benchmarks
- `bench/llm_bench.py` (Layer A baseline suite)
- Report p50/p95/p99 and per-token timing where relevant

### Tests
- Golden vectors for tokenizer (fixed inputs/outputs)
- Deterministic sampling with seeded RNG
- Cosine scoring parity vs NumPy reference

### Baseline Performance (Python/NumPy)
- **sample_token**: 83-270µs per token (vocab 32K-128K)
- **cosine_exact/candidates**: 0.25-1.3ms per query
- **kv_cache**: ~1µs per update
- **tokenizer**: 20-624µs per text (128-4096 chars)

---

## Phase 3: Zig LLM Kernels + Python Bindings (Current)

### Zig
- Fused `sample_token_f32` (penalty + top-k + top-p + RNG)
- Cosine scoring kernel (exact + candidate list)
- KV cache ring write kernel
- Tokenizer helper(s): optional byte/BPE merge loop

### Python
- ctypes bindings for each kernel
- RNG seed control for deterministic benchmarks
- Input validation (contiguous float32 buffers)

### Benchmarks
- Add Zig implementations to `bench/llm_bench.py`
- Report vs Python baseline and vs NumPy where applicable

### Tests
- Zig unit tests for each new kernel
- Python parity tests vs baseline outputs
- Stress tests for large vocab/corpus sizes

---

## Phase 4: Decode-Step Macrobench (Layer B)

### Scope
- Measure TPOT for a single decode step with clear boundaries:
  - sampling only (logits already on CPU)
  - sampling + transfer (GPU->CPU) if applicable

### Benchmarks
- `bench/decode_step.py` (new)
- Report p50/p95/p99 TPOT and batch scaling

### Tests
- Functional correctness vs baseline sampling
- Determinism with fixed RNG seed

---

## Phase 5: Serving Benchmarks (Layer C)

### Scope
- End-to-end TTFT, TPOT, TPS under concurrent load
- Integration with LLMPerf or equivalent harness

### Benchmarks
- `bench/serving/` harness configs
- Report TTFT/TPOT/TPS with explicit metric definitions

### Tests
- Config validation
- Smoke test for harness execution

---

## Phase 6: Type Coverage, SIMD Dispatch, Packaging

### Zig
- f32, i64, i32, u8 variants
- Optional SIMD width dispatch (AVX2/AVX-512/NEON)

### Python
- Dtype-based dispatch
- Packaging for wheels (cibuildwheel)

### Benchmarks
- Dtype-specific regression baselines
- SIMD dispatch comparisons

### Tests
- Cross-dtype parity checks
- Multi-thread correctness for supported kernels

---

## Phase 7: Ecosystem Integration

### Python
- Optional NumPy ufuncs
- Pandas/Polars integration hooks

### Benchmarks
- End-to-end data frame operations (aggregation, normalization)

### Tests
- Ufunc correctness
- Arrow round-trip validation

---

## Benchmark and Test Matrix

| Layer | Script | Purpose | Output |
| --- | --- | --- | --- |
| A | `bench/llm_bench.py` | Microkernels | p50/p95/p99, per-token |
| B | `bench/decode_step.py` | TPOT macrobench | p50/p95/p99 TPOT |
| C | `bench/serving/*` | TTFT/TPOT/TPS | latency + throughput |
| Core | `bench/bench.py` | Numeric kernels | ms + speedups |

## Regression Policy

- Keep JSON baselines per machine signature: CPU model, OS, Python, build flags.
- Strict gating only on pinned hardware; CI uses relaxed thresholds.
- Always report both single-thread and multi-thread results.

## Timeline (Rough)

| Phase | Target |
| --- | --- |
| Phase 0 | Done |
| Phase 1 | Done |
| Phase 1.5 | Done |
| Phase 2 | Done |
| Phase 3 | Q2 2026 |
| Phase 4 | Q2 2026 |
| Phase 5 | Q2-Q3 2026 |
| Phase 6 | Q3 2026 |
| Phase 7 | Q4 2026 |
