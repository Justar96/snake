"""
Unit tests for zigops Python bindings.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from snake import sum_sq, sum_sq_mt, dot, clip, argmax


class TestSumSq:
    def test_basic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = sum_sq(a)
        expected = 1 + 4 + 9 + 16  # 30
        assert abs(result - expected) < 1e-10

    def test_empty(self):
        a = np.array([], dtype=np.float64)
        result = sum_sq(a)
        assert result == 0.0

    def test_single(self):
        a = np.array([5.0])
        result = sum_sq(a)
        assert abs(result - 25.0) < 1e-10

    def test_large(self):
        rng = np.random.default_rng(42)
        a = rng.random(1_000_000)
        result = sum_sq(a)
        expected = float(a @ a)
        assert abs(result - expected) / expected < 1e-10

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(10_000)
        result = sum_sq(a)
        expected = float(np.dot(a, a))
        assert abs(result - expected) / expected < 1e-10


class TestSumSqMt:
    def test_matches_single_thread(self):
        rng = np.random.default_rng(42)
        a = rng.random(2_000_000)  # Large enough to use threads
        single = sum_sq(a)
        multi = sum_sq_mt(a)
        assert abs(single - multi) / single < 1e-10

    def test_explicit_thread_count(self):
        rng = np.random.default_rng(42)
        a = rng.random(2_000_000)
        result_2 = sum_sq_mt(a, n_threads=2)
        result_4 = sum_sq_mt(a, n_threads=4)
        expected = float(a @ a)
        assert abs(result_2 - expected) / expected < 1e-10
        assert abs(result_4 - expected) / expected < 1e-10


class TestDot:
    def test_basic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])
        result = dot(a, b)
        expected = 1*2 + 2*3 + 3*4 + 4*5  # 40
        assert abs(result - expected) < 1e-10

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(10_000)
        b = rng.random(10_000)
        result = dot(a, b)
        expected = float(np.dot(a, b))
        assert abs(result - expected) / abs(expected) < 1e-10

    def test_length_mismatch(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            dot(a, b)


class TestClip:
    def test_basic(self):
        a = np.array([-1.0, 0.5, 1.5, 3.0])
        result = clip(a, 0.0, 1.0)
        expected = np.array([0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_in_place(self):
        a = np.array([-1.0, 0.5, 1.5, 3.0])
        result = clip(a, 0.0, 1.0)
        # Result should be the same array (modified in place)
        # Note: due to ascontiguousarray, may be a copy if not already contiguous
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0, 1.0])

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(1000) * 2 - 0.5  # Range [-0.5, 1.5]
        a_copy = a.copy()
        result = clip(a_copy, 0.0, 1.0)
        expected = np.clip(a, 0.0, 1.0)
        np.testing.assert_array_almost_equal(result, expected)


class TestArgmax:
    def test_basic(self):
        a = np.array([1.0, 4.0, 2.0, 3.0])
        result = argmax(a)
        assert result == 1

    def test_first_element(self):
        a = np.array([5.0, 1.0, 2.0, 3.0])
        result = argmax(a)
        assert result == 0

    def test_last_element(self):
        a = np.array([1.0, 2.0, 3.0, 5.0])
        result = argmax(a)
        assert result == 3

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(10_000)
        result = argmax(a)
        expected = int(np.argmax(a))
        assert result == expected

    def test_ties_first_wins(self):
        # When there are ties, first occurrence wins
        a = np.array([1.0, 5.0, 5.0, 2.0])
        result = argmax(a)
        assert result == 1


class TestDtypeConversion:
    def test_float32_converted(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = sum_sq(a)
        expected = 14.0
        assert abs(result - expected) < 1e-5

    def test_int_converted(self):
        a = np.array([1, 2, 3, 4], dtype=np.int32)
        result = sum_sq(a)
        expected = 30.0
        assert abs(result - expected) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
