"""
snake - High-performance numeric kernels written in Zig for Python.

Bypasses the GIL with native SIMD and multi-threaded operations.
"""

from snake._core import (
    argmax,
    clip,
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
    softmax_mt,
    sum_sq,
    sum_sq_mt,
    variance,
)

__version__ = "0.1.0"
__all__ = [
    "argmax",
    "clip",
    "cumsum",
    "dot",
    "gelu",
    "histogram",
    "normalize",
    "relu",
    "rolling_sum",
    "saxpy",
    "scale",
    "softmax",
    "softmax_mt",
    "sum_sq",
    "sum_sq_mt",
    "variance",
]
