#!/usr/bin/env python3
"""
Modern CLI styling utilities for benchmark output.

Features:
- Unicode box drawing with proper spacing
- ANSI colors with graceful fallback
- Clean single-line progress indicators
- Formatted tables with alignment
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    """ANSI color codes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


# Check if colors should be enabled
def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


COLORS_ENABLED = _colors_enabled()


def c(color: Color, text: str) -> str:
    """Apply color to text if colors are enabled."""
    if not COLORS_ENABLED:
        return text
    return f"{color.value}{text}{Color.RESET.value}"


def bold(text: str) -> str:
    """Make text bold."""
    return c(Color.BOLD, text)


def dim(text: str) -> str:
    """Make text dim."""
    return c(Color.DIM, text)


# Box drawing characters (single line, rounded)
BOX_TL = "╭"
BOX_TR = "╮"
BOX_BL = "╰"
BOX_BR = "╯"
BOX_H = "─"
BOX_V = "│"
BOX_L = "├"
BOX_R = "┤"


@dataclass
class BoxStyle:
    """Box styling configuration."""

    width: int = 70
    title_color: Color = Color.BRIGHT_CYAN
    border_color: Color = Color.BRIGHT_BLACK
    content_color: Color = Color.WHITE


def box_top(title: str = "", style: BoxStyle | None = None) -> str:
    """Create top of a box with optional title."""
    s = style or BoxStyle()
    border = c(s.border_color, BOX_TL + BOX_H)

    if title:
        title_text = c(s.title_color, f" {title} ")
        visible_len = len(f" {title} ")
        remaining = s.width - 2 - visible_len
        border += title_text + c(s.border_color, BOX_H * remaining + BOX_TR)
    else:
        border += c(s.border_color, BOX_H * (s.width - 2) + BOX_TR)

    return border


def box_row(
    content: str = "", style: BoxStyle | None = None, align: str = "left"
) -> str:
    """Create a row inside a box."""
    s = style or BoxStyle()

    # Calculate visible length (without ANSI codes)
    visible_len = len(_strip_ansi(content))
    padding = s.width - 4 - visible_len

    if align == "center":
        left_pad = padding // 2
        right_pad = padding - left_pad
        padded = " " * left_pad + content + " " * right_pad
    elif align == "right":
        padded = " " * padding + content
    else:
        padded = content + " " * max(0, padding)

    return c(s.border_color, BOX_V) + " " + padded + " " + c(s.border_color, BOX_V)


def box_divider(style: BoxStyle | None = None) -> str:
    """Create a horizontal divider inside a box."""
    s = style or BoxStyle()
    return c(s.border_color, BOX_L + BOX_H * (s.width - 2) + BOX_R)


def box_bottom(style: BoxStyle | None = None) -> str:
    """Create bottom of a box."""
    s = style or BoxStyle()
    return c(s.border_color, BOX_BL + BOX_H * (s.width - 2) + BOX_BR)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


def section_header(title: str, subtitle: str = "") -> None:
    """Print a section header with modern styling."""
    style = BoxStyle()
    print()
    print(box_top(title, style))
    if subtitle:
        print(box_row(c(Color.DIM, subtitle), style))
    print(box_bottom(style))
    print()


def info_line(label: str, value: str, unit: str = "") -> None:
    """Print a single info line with label and value."""
    label_str = c(Color.DIM, f"{label}:")
    value_str = c(Color.WHITE, str(value))
    unit_str = c(Color.DIM, unit) if unit else ""
    print(f"  {label_str} {value_str}{unit_str}")


def success(msg: str) -> None:
    """Print a success message."""
    print(f"  {c(Color.GREEN, '✓')} {msg}")


def warning(msg: str) -> None:
    """Print a warning message."""
    print(f"  {c(Color.YELLOW, '⚠')} {c(Color.YELLOW, msg)}")


def error(msg: str) -> None:
    """Print an error message."""
    print(f"  {c(Color.RED, '✗')} {c(Color.RED, msg)}")


def progress_dot() -> None:
    """Print a progress dot."""
    print(c(Color.DIM, "."), end="", flush=True)


class Table:
    """Modern table formatter with clean styling."""

    def __init__(self, columns: list[tuple[str, int, str]]):
        """
        Initialize table with columns.

        Args:
            columns: List of (header, width, alignment) tuples.
                     alignment: 'left', 'right', or 'center'
        """
        self.columns = columns
        self.style = BoxStyle()

    def _format_cell(self, value: str, width: int, align: str) -> str:
        """Format a cell value with proper width and alignment."""
        visible_len = len(_strip_ansi(value))
        padding = width - visible_len

        if padding < 0:
            # Truncate if too long
            return value[:width]

        if align == "right":
            return " " * padding + value
        elif align == "center":
            left = padding // 2
            right = padding - left
            return " " * left + value + " " * right
        else:
            return value + " " * padding

    def print_header(self) -> None:
        """Print table header."""
        cells = []
        for header, width, align in self.columns:
            cells.append(self._format_cell(c(Color.DIM, header), width, align))
        print("  " + "  ".join(cells))

        # Underline
        underline_parts = []
        for _, width, _ in self.columns:
            underline_parts.append(c(Color.BRIGHT_BLACK, "─" * width))
        print("  " + "  ".join(underline_parts))

    def print_row(self, values: list[str]) -> None:
        """Print a data row."""
        cells = []
        for i, (_, width, align) in enumerate(self.columns):
            value = values[i] if i < len(values) else ""
            cells.append(self._format_cell(value, width, align))
        print("  " + "  ".join(cells))

    def print_separator(self) -> None:
        """Print a light separator."""
        sep_parts = []
        for _, width, _ in self.columns:
            sep_parts.append(c(Color.BRIGHT_BLACK, "·" * width))
        print("  " + "  ".join(sep_parts))


def format_time_ms(seconds: float) -> str:
    """Format time in milliseconds with color coding."""
    ms = seconds * 1000
    if ms < 1:
        return c(Color.GREEN, f"{ms:.3f}ms")
    elif ms < 10:
        return c(Color.BRIGHT_GREEN, f"{ms:.3f}ms")
    elif ms < 100:
        return c(Color.YELLOW, f"{ms:.2f}ms")
    else:
        return c(Color.RED, f"{ms:.1f}ms")


def format_time_us(ns: float) -> str:
    """Format time in microseconds from nanoseconds."""
    us = ns / 1000
    if us < 10:
        return c(Color.GREEN, f"{us:.2f}μs")
    elif us < 100:
        return c(Color.BRIGHT_GREEN, f"{us:.2f}μs")
    elif us < 1000:
        return c(Color.YELLOW, f"{us:.1f}μs")
    else:
        return c(Color.RED, f"{us / 1000:.2f}ms")


def format_speedup(speedup: float) -> str:
    """Format speedup with color coding."""
    if speedup >= 2.0:
        return c(Color.GREEN, f"{speedup:.2f}×")
    elif speedup >= 1.0:
        return c(Color.BRIGHT_GREEN, f"{speedup:.2f}×")
    elif speedup >= 0.5:
        return c(Color.YELLOW, f"{speedup:.2f}×")
    else:
        return c(Color.RED, f"{speedup:.2f}×")


def format_number(n: int | float) -> str:
    """Format a number with thousands separators."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def format_size_mb(bytes_val: float) -> str:
    """Format bytes as MB."""
    return f"{bytes_val / 1024 / 1024:.1f} MB"


def banner(title: str, version: str = "") -> None:
    """Print a prominent banner."""
    style = BoxStyle(title_color=Color.BRIGHT_MAGENTA)
    print()
    print(box_top(style=style))
    print(box_row(c(Color.BOLD, title), style, align="center"))
    if version:
        print(box_row(c(Color.DIM, version), style, align="center"))
    print(box_bottom(style))


def completion_banner(msg: str = "Done!") -> None:
    """Print a completion banner."""
    style = BoxStyle(title_color=Color.GREEN, border_color=Color.GREEN)
    print()
    print(box_top(style=style))
    print(box_row(c(Color.GREEN, f"✓ {msg}"), style, align="center"))
    print(box_bottom(style))
    print()
