# snake 🐍⚡

> High-performance numeric kernels written in Zig for Python

**snake** provides SIMD-vectorized, GIL-free numeric operations callable from Python via ctypes. Designed to dramatically speed up tight numeric loops.

## Why?

- **Per-element loops are slow** — Python bytecode + object handling per iteration
- **GIL blocks CPU parallelism** — threads don't help for CPU-bound work
- **Zig solves both** — native SIMD, native threading, tiny shared library

## Quick Start

```bash
# Build
zig build -Doptimize=ReleaseFast

# Install
pip install -e ".[dev]"
```

```python
import numpy as np
from snake import sum_sq, sum_sq_mt, dot, clip, argmax

a = np.random.random(10_000_000)
result = sum_sq(a)        # SIMD single-thread
result = sum_sq_mt(a)     # Multi-threaded
```

## Kernels

Core kernels:

- sum_sq, sum_sq_mt, dot, clip, argmax

Phase 1 kernels:

- normalize, scale, saxpy
- relu, gelu, softmax
- cumsum, rolling_sum
- variance, histogram

## Benchmarks

```bash
python bench/bench.py
```

LLM-oriented microbenchmarks (Layer A):

```bash
python bench/llm_bench.py
```

| Function    | vs Python loop | vs NumPy             |
| ----------- | -------------- | -------------------- |
| `sum_sq`    | ≥ 20× faster   | ~1.0× (parity)       |
| `sum_sq_mt` | ≥ 50× faster   | > 1× on most systems |

## Project Structure

```
snake/
├── build.zig        # Zig build config
├── src/snake.zig    # Core Zig kernels
├── python/snake/    # Python bindings
├── bench/           # Benchmarks
├── docs/            # Architecture & roadmap
└── tests/           # Unit tests
```

## License

MIT
