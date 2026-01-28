#!/usr/bin/env python3
"""
Benchmark suite for zigops.

Compares:
- Pure Python loop (slow baseline)
- NumPy native operations (expected fast)
- Zig single-threaded (should match or beat NumPy)
- Zig multi-threaded (should scale with cores)

Usage:
    # Quick run
    python bench/bench.py

    # Full pyperf calibration
    python -m pyperf system tune
    python bench/bench.py --rigorous
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from snake import sum_sq, sum_sq_mt, dot, argmax


def py_loop_sum_sq(a: np.ndarray) -> float:
    """Pure Python baseline - intentionally slow."""
    s = 0.0
    for x in a:
        s += x * x
    return s


def np_dot_sum_sq(a: np.ndarray) -> float:
    """NumPy: a @ a is equivalent to sum of squares."""
    return float(a @ a)


def benchmark_sum_sq(n: int = 10_000_000, rigorous: bool = False):
    """Run sum of squares benchmark."""
    print(f"\n{'='*60}")
    print(f"Sum of Squares Benchmark (n = {n:,})")
    print(f"{'='*60}\n")

    # Generate test data
    rng = np.random.default_rng(42)
    a = rng.random(n, dtype=np.float64)

    # Sanity check: all implementations should give same result
    ref = np_dot_sum_sq(a)
    zig_result = sum_sq(a)
    zig_mt_result = sum_sq_mt(a)

    rel_err_zig = abs(zig_result - ref) / ref
    rel_err_mt = abs(zig_mt_result - ref) / ref

    print(f"Reference (NumPy):     {ref:.6f}")
    print(f"Zig single-thread:     {zig_result:.6f} (rel err: {rel_err_zig:.2e})")
    print(f"Zig multi-thread:      {zig_mt_result:.6f} (rel err: {rel_err_mt:.2e})")

    if rel_err_zig > 1e-9 or rel_err_mt > 1e-9:
        print("\n⚠️  WARNING: Relative error too high!")
        return

    print("\n✓ Sanity check passed\n")

    # Timing
    iterations = 100 if rigorous else 10

    results = {}

    # Python loop (only for small n, it's too slow otherwise)
    if n <= 100_000:
        start = time.perf_counter()
        for _ in range(max(1, iterations // 10)):
            py_loop_sum_sq(a)
        elapsed = time.perf_counter() - start
        results["python_loop"] = elapsed / max(1, iterations // 10)
    else:
        # Extrapolate from smaller sample
        small_a = a[:10_000]
        start = time.perf_counter()
        for _ in range(10):
            py_loop_sum_sq(small_a)
        elapsed = time.perf_counter() - start
        results["python_loop"] = (elapsed / 10) * (n / 10_000)
        print(f"(Python loop extrapolated from n=10,000)")

    # NumPy
    start = time.perf_counter()
    for _ in range(iterations):
        np_dot_sum_sq(a)
    elapsed = time.perf_counter() - start
    results["numpy_dot"] = elapsed / iterations

    # Zig single-thread
    start = time.perf_counter()
    for _ in range(iterations):
        sum_sq(a)
    elapsed = time.perf_counter() - start
    results["zig_sum_sq"] = elapsed / iterations

    # Zig multi-thread
    start = time.perf_counter()
    for _ in range(iterations):
        sum_sq_mt(a)
    elapsed = time.perf_counter() - start
    results["zig_sum_sq_mt"] = elapsed / iterations

    # Print results
    print(f"{'Method':<20} {'Time (ms)':<12} {'vs Python':<12} {'vs NumPy':<12}")
    print("-" * 56)

    py_time = results["python_loop"]
    np_time = results["numpy_dot"]

    for name, t in results.items():
        vs_py = py_time / t if t > 0 else float("inf")
        vs_np = np_time / t if t > 0 else float("inf")
        print(f"{name:<20} {t*1000:>10.3f}ms {vs_py:>10.1f}× {vs_np:>10.2f}×")


def benchmark_dot(n: int = 10_000_000, rigorous: bool = False):
    """Run dot product benchmark."""
    print(f"\n{'='*60}")
    print(f"Dot Product Benchmark (n = {n:,})")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(42)
    a = rng.random(n, dtype=np.float64)
    b = rng.random(n, dtype=np.float64)

    ref = float(np.dot(a, b))
    zig_result = dot(a, b)

    rel_err = abs(zig_result - ref) / abs(ref)
    print(f"Reference (NumPy):     {ref:.6f}")
    print(f"Zig dot:               {zig_result:.6f} (rel err: {rel_err:.2e})")

    iterations = 100 if rigorous else 10

    # NumPy
    start = time.perf_counter()
    for _ in range(iterations):
        np.dot(a, b)
    np_time = (time.perf_counter() - start) / iterations

    # Zig
    start = time.perf_counter()
    for _ in range(iterations):
        dot(a, b)
    zig_time = (time.perf_counter() - start) / iterations

    print(f"\n{'Method':<20} {'Time (ms)':<12} {'vs NumPy':<12}")
    print("-" * 44)
    print(f"{'numpy_dot':<20} {np_time*1000:>10.3f}ms {1.0:>10.2f}×")
    print(f"{'zig_dot':<20} {zig_time*1000:>10.3f}ms {np_time/zig_time:>10.2f}×")


def benchmark_argmax(n: int = 10_000_000, rigorous: bool = False):
    """Run argmax benchmark."""
    print(f"\n{'='*60}")
    print(f"Argmax Benchmark (n = {n:,})")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(42)
    a = rng.random(n, dtype=np.float64)

    ref = int(np.argmax(a))
    zig_result = argmax(a)

    print(f"Reference (NumPy):     {ref}")
    print(f"Zig argmax:            {zig_result}")
    assert ref == zig_result, "Results don't match!"

    iterations = 100 if rigorous else 10

    # NumPy
    start = time.perf_counter()
    for _ in range(iterations):
        np.argmax(a)
    np_time = (time.perf_counter() - start) / iterations

    # Zig
    start = time.perf_counter()
    for _ in range(iterations):
        argmax(a)
    zig_time = (time.perf_counter() - start) / iterations

    print(f"\n{'Method':<20} {'Time (ms)':<12} {'vs NumPy':<12}")
    print("-" * 44)
    print(f"{'numpy_argmax':<20} {np_time*1000:>10.3f}ms {1.0:>10.2f}×")
    print(f"{'zig_argmax':<20} {zig_time*1000:>10.3f}ms {np_time/zig_time:>10.2f}×")


def main():
    parser = argparse.ArgumentParser(description="zigops benchmarks")
    parser.add_argument("-n", type=int, default=10_000_000, help="Array size")
    parser.add_argument("--rigorous", action="store_true", help="More iterations")
    args = parser.parse_args()

    print("=" * 60)
    print("zigops Benchmark Suite")
    print("=" * 60)
    print(f"NumPy version: {np.__version__}")
    print(f"Array size: {args.n:,} float64 elements ({args.n * 8 / 1024 / 1024:.1f} MB)")

    benchmark_sum_sq(args.n, args.rigorous)
    benchmark_dot(args.n, args.rigorous)
    benchmark_argmax(args.n, args.rigorous)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
