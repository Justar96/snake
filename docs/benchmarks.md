# Benchmark Methodology

## Goals

Establish **clear, measurable, and fair** performance targets that demonstrate zigops' value proposition.

## Baselines

### 1. Pure Python Loop (Slow Baseline)

```python
def py_loop_sum_sq(a):
    s = 0.0
    for x in a:
        s += x * x
    return s
```

This is intentionally slow — it represents what a naive Python developer might write before reaching for NumPy.

### 2. NumPy (Fast Baseline)

```python
def np_sum_sq(a):
    return float(a @ a)  # or np.dot(a, a)
```

NumPy's BLAS-backed implementation. This is our "production quality" baseline.

## Target Metrics

### Primary Targets

| Metric                   | Target                | Rationale                              |
| ------------------------ | --------------------- | -------------------------------------- |
| vs Python loop           | ≥ 20× faster          | Obvious win for anyone not using NumPy |
| vs NumPy (single-thread) | ≥ 0.8× (within 1.25×) | Competitive with highly optimized BLAS |
| vs NumPy (multi-thread)  | ≥ 1.5× on 4+ cores    | Demonstrate parallelism value          |

### Secondary Metrics

| Metric           | Target                 | Notes                                    |
| ---------------- | ---------------------- | ---------------------------------------- |
| Call overhead    | < 1µs                  | For small arrays, overhead dominates     |
| Crossover point  | n ≈ 1,000              | Below this, ctypes overhead > savings    |
| Memory bandwidth | Near theoretical limit | For large arrays, should be memory-bound |

## Test Configurations

### Array Sizes

| Size   | Elements    | Memory | Purpose                |
| ------ | ----------- | ------ | ---------------------- |
| Tiny   | 100         | 800 B  | Overhead measurement   |
| Small  | 10,000      | 80 KB  | L2 cache resident      |
| Medium | 1,000,000   | 8 MB   | L3 cache resident      |
| Large  | 10,000,000  | 80 MB  | Main memory            |
| Huge   | 100,000,000 | 800 MB | Memory bandwidth bound |

### Data Types

Currently: `float64` only

Future: `float32`, `int64`, `int32`

## Methodology

### Warmup

- 3-5 warmup iterations before timing
- Ensures JIT compilation (Python) and cache warming

### Timing

- Use `time.perf_counter()` for wall-clock time
- Minimum 10 iterations for quick runs
- 100+ iterations for rigorous benchmarks

### Statistical Rigor

For publication-quality results, use `pyperf`:

```bash
python -m pyperf system tune  # Disable CPU frequency scaling
python -m pyperf timeit -s "setup" "statement"
```

This handles:

- Multiple processes to avoid GC variance
- Automatic iteration count calibration
- Outlier detection and removal
- Confidence intervals

## Known Confounding Factors

### NumPy BLAS Configuration

NumPy may use different BLAS backends:

- **OpenBLAS** (common on Linux)
- **MKL** (Intel, fastest on Intel CPUs)
- **Accelerate** (macOS, optimized for Apple Silicon)

Check with:

```python
import numpy as np
np.show_config()
```

### CPU Frequency Scaling

- Turbo boost can cause variance
- Thermal throttling affects sustained workloads
- Use `pyperf system tune` to stabilize

### Memory Allocation

- First call may trigger page faults
- Use preallocated arrays in benchmarks

### Cache Effects

- Array size relative to L1/L2/L3 matters
- Cold vs warm cache significantly affects results
- Document cache sizes for reproducibility

## Reporting Format

```
============================================================
zigops Benchmark Results
============================================================
System: Intel Core i7-12700K, 32GB DDR4-3200
NumPy: 1.26.4 (OpenBLAS)
Zig: 0.13.0 (ReleaseFast)
------------------------------------------------------------

Sum of Squares (n = 10,000,000)
------------------------------------------------------------
Method              Time (ms)    vs Python    vs NumPy
python_loop          842.31ms        1.0×        0.01×
numpy_dot              8.12ms      103.7×        1.00×
zig_sum_sq             7.89ms      106.8×        1.03×
zig_sum_sq_mt          2.34ms      360.0×        3.47×

✓ Meets targets: Python ≥20× ✓, NumPy ≥0.8× ✓, MT ≥1.5× ✓
```

## Reproducing Results

```bash
# Build with optimizations
zig build -Doptimize=ReleaseFast

# Install Python dependencies
pip install numpy pyperf

# Run benchmarks
python bench/bench.py

# For rigorous results
sudo python -m pyperf system tune
python bench/bench.py --rigorous
```
