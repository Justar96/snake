# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**snake** is a Python library providing SIMD-vectorized, GIL-free numeric operations via Zig kernels. NumPy arrays are passed directly to Zig via ctypes pointers for zero-copy access.

## Build and Development Commands

```bash
# Build the shared library (required before Python usage)
zig build -Doptimize=ReleaseFast

# Run Zig unit tests
zig build test

# Install Python package in development mode
pip install -e ".[dev]"

# Run Python tests
pytest tests/

# Run a single Python test
pytest tests/test_zigops.py::TestSumSq::test_basic -v

# Run benchmarks
python bench/bench.py

# Lint Python code
ruff check python/
ruff format python/
```

## Architecture

### Data Flow
```
numpy.ndarray → ctypes.POINTER(c_double) → Zig kernel (SIMD) → return value
```

The GIL is automatically released during ctypes foreign function calls.

### Key Components

- **src/zigops.zig**: Core Zig kernels using `@Vector(4, f64)` for explicit SIMD. Exports C ABI functions (`sum_sq_f64`, `dot_f64`, `clip_f64`, `argmax_f64`, `sum_sq_f64_mt`).

- **python/snake/_core.py**: ctypes bindings that load `libsnake.so`, define function signatures, and convert NumPy arrays to `(ptr, len)` tuples via `_as_f64_ptr()`.

- **build.zig**: Builds dynamic library to `zig-out/lib/libsnake.so` and defines the `test` step.

### Multi-threading

`sum_sq_f64_mt` spawns up to 64 worker threads, each computing a partial sum on a chunk. Falls back to single-threaded for arrays < 1M elements.

### SIMD Pattern

All kernels follow the same structure:
1. SIMD loop processing 4 elements per iteration via `@Vector(4, f64)`
2. Scalar tail loop for remaining 0-3 elements
3. `@reduce(.Add, acc)` for horizontal reduction
