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


def _as_f64_ptr(arr: NDArray) -> tuple:
    """Convert numpy array to (pointer, length, array) tuple."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return ptr, arr.size, arr


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
