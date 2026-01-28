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

# Install Python dependencies (uses uv)
uv sync --group dev

# Run Python tests
uv run pytest tests/

# Run a single Python test
uv run pytest tests/test_snake.py::test_sum_sq -v

# Run benchmarks
uv run python bench/bench.py
uv run python bench/llm_bench.py      # LLM-oriented microbenchmarks
uv run python bench/micro_reduce.py   # Reduction kernel benchmarks

# Lint and format Python code
uv run ruff check python/
uv run ruff format python/

# Type check Python code
uv run ty check python/
```

## Code Style

- **File size:** Aim to keep files under ~700 LOC; guideline only (not a hard guardrail). Split/refactor when it improves clarity or testability.
- **No V2 copies:** Keep files concise; extract helpers instead of duplicating with "V2" suffixes. Use existing patterns in the codebase.
- **Linting:** Run `uv run ruff check python/` before commits.
- **Formatting:** Run `uv run ruff format python/` to auto-format.
- **Type checking:** Run `uv run ty check python/` to verify types.
- **Comments:** Add brief code comments for tricky or non-obvious logic.

## Architecture

### Data Flow

```
numpy.ndarray → ctypes.POINTER(c_double) → Zig kernel (SIMD) → return value
```

The GIL is automatically released during ctypes foreign function calls.

### Zig Module Structure

```
src/
├── snake.zig           # Root module with C ABI exports
├── simd/
│   ├── simd.zig        # SIMD type aliases (Vec4) and helpers
│   └── reduce.zig      # Core SIMD reduction patterns (sum, sumSq, dot)
├── kernels/
│   ├── kernels.zig     # Re-exports all kernel modules
│   ├── reductions.zig  # sum_sq, dot, variance, argmax
│   ├── transforms.zig  # clip, scale, normalize, saxpy
│   ├── activations.zig # relu, gelu, softmax
│   ├── prefix.zig      # cumsum, rolling_sum
│   └── histogram.zig   # histogram
└── threading/
    └── threading.zig   # Multi-threaded variants (sum_sq_mt, softmax_mt)
```

### Key Components

- **src/snake.zig**: Root module that exports C ABI functions. All kernels are thin wrappers delegating to the `kernels/` and `threading/` submodules.

- **src/simd/reduce.zig**: Core SIMD reduction helpers using `@Vector(4, f64)`. Handles memory alignment and loop unrolling (2x unroll for ILP).

- **python/snake/\_core.py**: ctypes bindings that load `libsnake.so`, define function signatures, and convert NumPy arrays to `(ptr, len)` tuples via `_as_f64_ptr()`.

- **build.zig**: Builds dynamic library to `zig-out/lib/libsnake.so` and defines the `test` step.

### Multi-threading

`sum_sq_f64_mt` and `softmax_f64_mt` spawn worker threads, each computing a partial result on a chunk. Falls back to single-threaded for small arrays.

### SIMD Pattern

Reduction kernels in `src/simd/reduce.zig` follow an optimized structure:

1. **Alignment prefix**: Scalar loop to reach vector-aligned memory boundary
2. **Unrolled SIMD loop**: Process 8 elements per iteration (2×Vec4) using `@mulAdd` for FMA
3. **SIMD cleanup**: Process remaining 4-element chunks
4. **Horizontal reduce**: `@reduce(.Add, acc)` to collapse vector to scalar
5. **Scalar tail**: Handle remaining 0-3 elements
