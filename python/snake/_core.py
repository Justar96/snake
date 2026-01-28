"""
ctypes bindings for the Zig shared library.
"""

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _find_library() -> Path:
    """Locate the snake shared library."""
    candidates = [
        Path(__file__).parent.parent.parent / "zig-out" / "lib" / "libsnake.so",
        Path(__file__).parent.parent.parent / "zig-out" / "lib" / "libsnake.dylib",
        Path(__file__).parent.parent.parent / "zig-out" / "lib" / "snake.dll",
        Path(__file__).parent / "libsnake.so",
        Path(__file__).parent / "libsnake.dylib",
        Path(__file__).parent / "snake.dll",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise RuntimeError(
        "Could not find snake shared library. "
        "Run `zig build -Doptimize=ReleaseFast` first."
    )


_lib = ctypes.CDLL(str(_find_library()))

# Function signatures
_lib.simd_f64_lanes.argtypes = []
_lib.simd_f64_lanes.restype = ctypes.c_uint32

_lib.simd_f32_lanes.argtypes = []
_lib.simd_f32_lanes.restype = ctypes.c_uint32

_lib.sum_sq_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.sum_sq_f64.restype = ctypes.c_double

_lib.sum_sq_f64_mt.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_uint32,
]
_lib.sum_sq_f64_mt.restype = ctypes.c_double

_lib.dot_f64.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
_lib.dot_f64.restype = ctypes.c_double

_lib.clip_f64.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_double,
]
_lib.clip_f64.restype = None

_lib.argmax_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.argmax_f64.restype = ctypes.c_size_t

# New Phase 1 kernels
_lib.normalize_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.normalize_f64.restype = None

_lib.scale_f64.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
]
_lib.scale_f64.restype = None

_lib.saxpy_f64.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
]
_lib.saxpy_f64.restype = None

_lib.relu_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.relu_f64.restype = None

_lib.gelu_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.gelu_f64.restype = None

_lib.softmax_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.softmax_f64.restype = None

_lib.softmax_f64_mt.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_uint32,
]
_lib.softmax_f64_mt.restype = None

_lib.cumsum_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.cumsum_f64.restype = None

_lib.rolling_sum_f64.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.rolling_sum_f64.restype = None

_lib.variance_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.variance_f64.restype = ctypes.c_double

_lib.histogram_f64.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_double,
]
_lib.histogram_f64.restype = None


def _as_f64_ptr(arr: NDArray) -> tuple:
    """Convert numpy array to (pointer, length, array) tuple."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return ptr, arr.size, arr


def simd_lanes() -> tuple[int, int]:
    """Return SIMD lane counts (f64 lanes, f32 lanes) for this build."""
    return int(_lib.simd_f64_lanes()), int(_lib.simd_f32_lanes())


def sum_sq(a: NDArray) -> float:
    """Compute sum of squares: Σ(x²) using SIMD."""
    ptr, size, _ = _as_f64_ptr(a)
    return float(_lib.sum_sq_f64(ptr, size))


def sum_sq_mt(a: NDArray, n_threads: int = 0) -> float:
    """Multi-threaded sum of squares (0 = auto-detect cores)."""
    ptr, size, _ = _as_f64_ptr(a)
    return float(_lib.sum_sq_f64_mt(ptr, size, int(n_threads)))


def dot(a: NDArray, b: NDArray) -> float:
    """Compute dot product: Σ(a · b)."""
    ptr_a, size_a, _ = _as_f64_ptr(a)
    ptr_b, size_b, _ = _as_f64_ptr(b)
    if size_a != size_b:
        raise ValueError(f"Arrays must have same length: {size_a} != {size_b}")
    return float(_lib.dot_f64(ptr_a, ptr_b, size_a))


def clip(a: NDArray, lo: float, hi: float) -> NDArray:
    """Clip values in-place to [lo, hi] range."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.clip_f64(ptr, a.size, float(lo), float(hi))
    return a


def argmax(a: NDArray) -> int:
    """Find index of maximum value."""
    ptr, size, _ = _as_f64_ptr(a)
    return int(_lib.argmax_f64(ptr, size))


# =============================================================================
# Phase 1 Kernels
# =============================================================================


def normalize(a: NDArray) -> NDArray:
    """L2 normalize in-place: x[i] /= sqrt(Σx²)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.normalize_f64(ptr, a.size)
    return a


def scale(a: NDArray, scalar: float) -> NDArray:
    """Multiply array by scalar in-place: x[i] *= s."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.scale_f64(ptr, a.size, float(scalar))
    return a


def saxpy(a: NDArray, b: NDArray, scalar: float) -> NDArray:
    """SAXPY: a[i] += scalar * b[i]. Modifies a in-place."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr_a = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptr_b, size_b, _ = _as_f64_ptr(b)
    if a.size != size_b:
        raise ValueError(f"Arrays must have same length: {a.size} != {size_b}")
    _lib.saxpy_f64(ptr_a, ptr_b, a.size, float(scalar))
    return a


def relu(a: NDArray) -> NDArray:
    """ReLU activation in-place: x[i] = max(0, x[i])."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.relu_f64(ptr, a.size)
    return a


def gelu(a: NDArray) -> NDArray:
    """GELU activation in-place using tanh approximation."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.gelu_f64(ptr, a.size)
    return a


def softmax(a: NDArray) -> NDArray:
    """Softmax activation in-place: exp(x[i]) / Σexp(x)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.softmax_f64(ptr, a.size)
    return a


def softmax_mt(a: NDArray, n_threads: int = 0) -> NDArray:
    """Multi-threaded softmax activation in-place (0 = auto-detect cores)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.softmax_f64_mt(ptr, a.size, int(n_threads))
    return a


def cumsum(a: NDArray) -> NDArray:
    """Cumulative sum (prefix sum) in-place."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.cumsum_f64(ptr, a.size)
    return a


def rolling_sum(a: NDArray, window: int) -> NDArray:
    """Rolling sum with fixed window size."""
    a_in = np.ascontiguousarray(a, dtype=np.float64)
    a_out = np.empty_like(a_in)
    ptr_in = a_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptr_out = a_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.rolling_sum_f64(ptr_in, ptr_out, a_in.size, int(window))
    return a_out


def variance(a: NDArray) -> float:
    """Population variance using Welford's algorithm."""
    ptr, size, _ = _as_f64_ptr(a)
    return float(_lib.variance_f64(ptr, size))


def histogram(a: NDArray, n_bins: int, min_val: float, max_val: float) -> NDArray:
    """Compute binned histogram. Returns array of bin counts."""
    ptr_a, size_a, _ = _as_f64_ptr(a)
    bins = np.zeros(n_bins, dtype=np.float64)
    ptr_bins = bins.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _lib.histogram_f64(ptr_a, size_a, ptr_bins, n_bins, float(min_val), float(max_val))
    return bins
