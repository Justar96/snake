#!/usr/bin/env python3
"""
Microbench for reduction kernels (sum_sq, dot).

Runs a tight loop on a fixed array size to compare NumPy vs snake.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure NumPy uses a single thread for fair single-thread comparisons.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from snake import dot, sum_sq


def _time_loop(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations


def microbench(n: int, iterations: int) -> None:
    rng = np.random.default_rng(42)
    a = rng.random(n, dtype=np.float64)
    b = rng.random(n, dtype=np.float64)

    # Warm-up
    sum_sq(a)
    dot(a, b)

    np_sum_sq = _time_loop(lambda: float(a @ a), iterations)
    zig_sum_sq = _time_loop(lambda: sum_sq(a), iterations)

    np_dot = _time_loop(lambda: float(np.dot(a, b)), iterations)
    zig_dot = _time_loop(lambda: dot(a, b), iterations)

    print(f"n={n:,}  iterations={iterations}")
    print("sum_sq")
    print(f"  numpy: {np_sum_sq*1e3:.3f} ms")
    print(f"  zig:   {zig_sum_sq*1e3:.3f} ms  (speedup {np_sum_sq/zig_sum_sq:.2f}x)")
    print("dot")
    print(f"  numpy: {np_dot*1e3:.3f} ms")
    print(f"  zig:   {zig_dot*1e3:.3f} ms  (speedup {np_dot/zig_dot:.2f}x)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbench reduction kernels")
    parser.add_argument("--n", type=int, default=10_000_000)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    microbench(args.n, args.iterations)


if __name__ == "__main__":
    main()
