#!/usr/bin/env python3
"""
Benchmark suite for snake.

Compares:
- Pure Python loop (slow baseline)
- NumPy native operations (expected fast)
- Zig single-threaded (should match or beat NumPy)
- Zig multi-threaded (should scale with cores)
- Phase 1 kernels vs NumPy baselines

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

from snake import (
    argmax,
    cumsum,
    dot,
    gelu,
    histogram,
    normalize,
    relu,
    rolling_sum,
    saxpy,
    scale,
    softmax,
    sum_sq,
    sum_sq_mt,
    variance,
)

from cli_style import (
    Table,
    banner,
    c,
    Color,
    completion_banner,
    error,
    format_number,
    format_size_mb,
    format_speedup,
    format_time_ms,
    info_line,
    section_header,
    success,
    warning,
)


def py_loop_sum_sq(a: np.ndarray) -> float:
    """Pure Python baseline - intentionally slow."""
    s = 0.0
    for x in a:
        s += x * x
    return s


def np_dot_sum_sq(a: np.ndarray) -> float:
    """NumPy: a @ a is equivalent to sum of squares."""
    return float(a @ a)


def np_normalize(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a)
    if norm != 0.0:
        a /= norm
    return a


def np_scale(a: np.ndarray, scalar: float) -> np.ndarray:
    a *= scalar
    return a


def np_saxpy(a: np.ndarray, b: np.ndarray, scalar: float) -> np.ndarray:
    a += scalar * b
    return a


def np_relu(a: np.ndarray) -> np.ndarray:
    np.maximum(a, 0.0, out=a)
    return a


def np_gelu(a: np.ndarray) -> np.ndarray:
    sqrt_2_pi = 0.7978845608028654
    coeff = 0.044715
    a[:] = 0.5 * a * (1.0 + np.tanh(sqrt_2_pi * (a + coeff * a**3)))
    return a


def np_softmax(a: np.ndarray) -> np.ndarray:
    a -= np.max(a)
    np.exp(a, out=a)
    denom = np.sum(a)
    if denom != 0.0:
        a /= denom
    return a


def np_cumsum_inplace(a: np.ndarray) -> np.ndarray:
    a[:] = np.cumsum(a)
    return a


def np_rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    window = min(window, a.size)
    if window == 0:
        return np.zeros_like(a)
    cumsum = np.cumsum(a)
    out = cumsum.copy()
    out[window:] = cumsum[window:] - cumsum[:-window]
    return out


def np_variance(a: np.ndarray) -> float:
    return float(np.var(a))


def np_histogram(
    a: np.ndarray, n_bins: int, min_val: float, max_val: float
) -> np.ndarray:
    counts, _ = np.histogram(a, bins=n_bins, range=(min_val, max_val))
    return counts.astype(np.float64)


def _time_loop(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations


def _bench_inplace(src: np.ndarray, iterations: int, fn) -> float:
    scratch = np.empty_like(src)

    def step() -> None:
        np.copyto(scratch, src)
        fn(scratch)

    return _time_loop(step, iterations)


def _bench_inplace_pair(
    src_a: np.ndarray, src_b: np.ndarray, iterations: int, fn
) -> float:
    scratch = np.empty_like(src_a)

    def step() -> None:
        np.copyto(scratch, src_a)
        fn(scratch, src_b)

    return _time_loop(step, iterations)


def benchmark_sum_sq(n: int = 10_000_000, rigorous: bool = False):
    """Run sum of squares benchmark."""
    section_header("Sum of Squares", f"n = {format_number(n)} elements")

    # Generate test data
    rng = np.random.default_rng(42)
    a = rng.random(n, dtype=np.float64)

    # Sanity check: all implementations should give same result
    ref = np_dot_sum_sq(a)
    zig_result = sum_sq(a)
    zig_mt_result = sum_sq_mt(a)

    rel_err_zig = abs(zig_result - ref) / ref
    rel_err_mt = abs(zig_mt_result - ref) / ref

    info_line("Reference (NumPy)", f"{ref:.6f}")
    info_line("Zig single-thread", f"{zig_result:.6f}", f"  rel err: {rel_err_zig:.2e}")
    info_line(
        "Zig multi-thread", f"{zig_mt_result:.6f}", f"  rel err: {rel_err_mt:.2e}"
    )
    print()

    if rel_err_zig > 1e-9 or rel_err_mt > 1e-9:
        warning("Relative error too high!")
        return

    success("Sanity check passed")
    print()

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
        print(c(Color.DIM, "  (Python loop extrapolated from n=10,000)"))

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
    table = Table(
        [
            ("Method", 16, "left"),
            ("Time", 12, "right"),
            ("vs Python", 10, "right"),
            ("vs NumPy", 10, "right"),
        ]
    )
    table.print_header()

    py_time = results["python_loop"]
    np_time = results["numpy_dot"]

    for name, t in results.items():
        vs_py = py_time / t if t > 0 else float("inf")
        vs_np = np_time / t if t > 0 else float("inf")
        table.print_row(
            [
                c(Color.CYAN, name),
                format_time_ms(t),
                format_speedup(vs_py),
                format_speedup(vs_np),
            ]
        )


def benchmark_dot(n: int = 10_000_000, rigorous: bool = False):
    """Run dot product benchmark."""
    section_header("Dot Product", f"n = {format_number(n)} elements")

    rng = np.random.default_rng(42)
    a = rng.random(n, dtype=np.float64)
    b = rng.random(n, dtype=np.float64)

    ref = float(np.dot(a, b))
    zig_result = dot(a, b)

    rel_err = abs(zig_result - ref) / abs(ref)
    info_line("Reference (NumPy)", f"{ref:.6f}")
    info_line("Zig dot", f"{zig_result:.6f}", f"  rel err: {rel_err:.2e}")
    print()

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

    table = Table(
        [
            ("Method", 16, "left"),
            ("Time", 12, "right"),
            ("vs NumPy", 10, "right"),
        ]
    )
    table.print_header()
    table.print_row(
        [c(Color.CYAN, "numpy_dot"), format_time_ms(np_time), format_speedup(1.0)]
    )
    table.print_row(
        [
            c(Color.CYAN, "zig_dot"),
            format_time_ms(zig_time),
            format_speedup(np_time / zig_time),
        ]
    )


def benchmark_argmax(n: int = 10_000_000, rigorous: bool = False):
    """Run argmax benchmark."""
    section_header("Argmax", f"n = {format_number(n)} elements")

    rng = np.random.default_rng(42)
    a = rng.random(n, dtype=np.float64)

    ref = int(np.argmax(a))
    zig_result = argmax(a)

    info_line("Reference (NumPy)", str(ref))
    info_line("Zig argmax", str(zig_result))
    print()

    assert ref == zig_result, "Results don't match!"
    success("Verification passed")
    print()

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

    table = Table(
        [
            ("Method", 16, "left"),
            ("Time", 12, "right"),
            ("vs NumPy", 10, "right"),
        ]
    )
    table.print_header()
    table.print_row(
        [c(Color.CYAN, "numpy_argmax"), format_time_ms(np_time), format_speedup(1.0)]
    )
    table.print_row(
        [
            c(Color.CYAN, "zig_argmax"),
            format_time_ms(zig_time),
            format_speedup(np_time / zig_time),
        ]
    )


def benchmark_phase1(n: int = 1_000_000, rigorous: bool = False):
    """Run Phase 1 kernel benchmarks."""
    section_header("Phase 1 Kernels", "Vectorized operations vs NumPy")

    rng = np.random.default_rng(42)
    n_core = min(n, 1_000_000)
    n_softmax = min(n, 200_000)
    window = 64
    n_bins = 128

    a = rng.standard_normal(n_core, dtype=np.float64)
    b = rng.standard_normal(n_core, dtype=np.float64)
    softmax_input = rng.standard_normal(n_softmax, dtype=np.float64)
    hist_input = rng.random(n_core, dtype=np.float64)

    iterations = 50 if rigorous else 10
    heavy_iters = max(1, iterations // 5)

    # Sanity checks
    checks = [
        ("normalize", np.allclose(normalize(a.copy()), np_normalize(a.copy()))),
        ("scale", np.allclose(scale(a.copy(), 2.0), np_scale(a.copy(), 2.0))),
        ("saxpy", np.allclose(saxpy(a.copy(), b, 2.0), np_saxpy(a.copy(), b, 2.0))),
        ("relu", np.allclose(relu(a.copy()), np_relu(a.copy()))),
        ("gelu", np.allclose(gelu(a.copy()), np_gelu(a.copy()), rtol=1e-6, atol=1e-6)),
        (
            "softmax",
            np.allclose(
                softmax(softmax_input.copy()),
                np_softmax(softmax_input.copy()),
                rtol=1e-6,
            ),
        ),
        ("cumsum", np.allclose(cumsum(a.copy()), np_cumsum_inplace(a.copy()))),
        ("rolling_sum", np.allclose(rolling_sum(a, window), np_rolling_sum(a, window))),
        ("variance", np.isclose(variance(a), np_variance(a))),
        (
            "histogram",
            np.allclose(
                histogram(hist_input, n_bins, 0.0, 1.0),
                np_histogram(hist_input, n_bins, 0.0, 1.0),
            ),
        ),
    ]

    failed = [name for name, passed in checks if not passed]
    if failed:
        for name in failed:
            warning(f"{name} mismatch")
        return

    success("All kernel sanity checks passed")
    print()

    info_line("Core array size", format_number(n_core))
    info_line("Softmax array size", format_number(n_softmax))
    info_line("Rolling window", str(window))
    info_line("Histogram bins", str(n_bins))
    print()

    table = Table(
        [
            ("Kernel", 14, "left"),
            ("NumPy", 12, "right"),
            ("Zig", 12, "right"),
            ("Speedup", 10, "right"),
        ]
    )
    table.print_header()

    rows = []

    np_time = _bench_inplace(a, iterations, np_normalize)
    zig_time = _bench_inplace(a, iterations, normalize)
    rows.append(("normalize", np_time, zig_time))

    np_time = _bench_inplace(a, iterations, lambda x: np_scale(x, 2.0))
    zig_time = _bench_inplace(a, iterations, lambda x: scale(x, 2.0))
    rows.append(("scale", np_time, zig_time))

    np_time = _bench_inplace_pair(a, b, iterations, lambda x, y: np_saxpy(x, y, 2.0))
    zig_time = _bench_inplace_pair(a, b, iterations, lambda x, y: saxpy(x, y, 2.0))
    rows.append(("saxpy", np_time, zig_time))

    np_time = _bench_inplace(a, iterations, np_relu)
    zig_time = _bench_inplace(a, iterations, relu)
    rows.append(("relu", np_time, zig_time))

    np_time = _bench_inplace(a, iterations, np_gelu)
    zig_time = _bench_inplace(a, iterations, gelu)
    rows.append(("gelu", np_time, zig_time))

    np_time = _bench_inplace(softmax_input, heavy_iters, np_softmax)
    zig_time = _bench_inplace(softmax_input, heavy_iters, softmax)
    rows.append(("softmax", np_time, zig_time))

    np_time = _bench_inplace(a, iterations, np_cumsum_inplace)
    zig_time = _bench_inplace(a, iterations, cumsum)
    rows.append(("cumsum", np_time, zig_time))

    np_time = _time_loop(lambda: np_rolling_sum(a, window), iterations)
    zig_time = _time_loop(lambda: rolling_sum(a, window), iterations)
    rows.append(("rolling_sum", np_time, zig_time))

    np_time = _time_loop(lambda: np_variance(a), iterations)
    zig_time = _time_loop(lambda: variance(a), iterations)
    rows.append(("variance", np_time, zig_time))

    np_time = _time_loop(lambda: np_histogram(hist_input, n_bins, 0.0, 1.0), iterations)
    zig_time = _time_loop(lambda: histogram(hist_input, n_bins, 0.0, 1.0), iterations)
    rows.append(("histogram", np_time, zig_time))

    for name, np_t, zig_t in rows:
        speedup = np_t / zig_t if zig_t > 0 else float("inf")
        table.print_row(
            [
                c(Color.CYAN, name),
                format_time_ms(np_t),
                format_time_ms(zig_t),
                format_speedup(speedup),
            ]
        )


def main():
    parser = argparse.ArgumentParser(description="snake benchmarks")
    parser.add_argument("-n", type=int, default=10_000_000, help="Array size")
    parser.add_argument("--rigorous", action="store_true", help="More iterations")
    parser.add_argument(
        "--skip-phase1", action="store_true", help="Skip Phase 1 kernel benchmarks"
    )
    args = parser.parse_args()

    banner("🐍 snake Benchmark Suite", f"NumPy {np.__version__}")
    print()
    info_line("Array size", f"{format_number(args.n)} float64 elements")
    info_line("Memory", format_size_mb(args.n * 8))
    info_line("Mode", "rigorous" if args.rigorous else "quick")

    benchmark_sum_sq(args.n, args.rigorous)
    benchmark_dot(args.n, args.rigorous)
    benchmark_argmax(args.n, args.rigorous)
    if not args.skip_phase1:
        benchmark_phase1(args.n, args.rigorous)

    completion_banner("Benchmark complete!")


if __name__ == "__main__":
    main()
