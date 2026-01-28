"""
snake - High-performance numeric kernels written in Zig for Python.

Bypasses the GIL with native SIMD and multi-threaded operations.
"""

from snake._core import (
    argmax,
    clip,
    dot,
    sum_sq,
    sum_sq_mt,
)

__version__ = "0.1.0"
__all__ = ["argmax", "clip", "dot", "sum_sq", "sum_sq_mt"]
