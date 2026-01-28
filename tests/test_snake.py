"""
Unit tests for snake Python bindings.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from snake import (
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
        expected = 1 * 2 + 2 * 3 + 3 * 4 + 4 * 5  # 40
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


class TestNormalize:
    def test_basic(self):
        a = np.array([3.0, 4.0])
        result = normalize(a.copy())
        # L2 norm = 5, normalized = [0.6, 0.8]
        np.testing.assert_array_almost_equal(result, [0.6, 0.8])

    def test_unit_vector(self):
        a = np.array([1.0, 0.0, 0.0])
        result = normalize(a.copy())
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(1000)
        result = normalize(a.copy())
        expected = a / np.linalg.norm(a)
        np.testing.assert_array_almost_equal(result, expected)


class TestScale:
    def test_basic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = scale(a.copy(), 2.0)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0, 8.0])

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(1000)
        scalar = 3.14
        result = scale(a.copy(), scalar)
        expected = a * scalar
        np.testing.assert_array_almost_equal(result, expected)


class TestSaxpy:
    def test_basic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 1.0, 1.0, 1.0])
        result = saxpy(a.copy(), b, 2.0)
        np.testing.assert_array_almost_equal(result, [3.0, 4.0, 5.0, 6.0])

    def test_length_mismatch(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            saxpy(a.copy(), b, 1.0)


class TestRelu:
    def test_basic(self):
        a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu(a.copy())
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(1000) * 2 - 1  # Range [-1, 1]
        result = relu(a.copy())
        expected = np.maximum(0, a)
        np.testing.assert_array_almost_equal(result, expected)


class TestGelu:
    def test_zero(self):
        a = np.array([0.0])
        result = gelu(a.copy())
        assert abs(result[0]) < 1e-6

    def test_positive(self):
        a = np.array([1.0])
        result = gelu(a.copy())
        # GELU(1) ≈ 0.841
        assert abs(result[0] - 0.841) < 0.01


class TestSoftmax:
    def test_basic(self):
        a = np.array([1.0, 2.0, 3.0])
        result = softmax(a.copy())
        # Sum should be 1
        assert abs(result.sum() - 1.0) < 1e-10
        # Values should be ordered
        assert result[0] < result[1] < result[2]

    def test_numerical_stability(self):
        # Large values shouldn't overflow
        a = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(a.copy())
        assert abs(result.sum() - 1.0) < 1e-10

    def test_multithreaded(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = softmax_mt(a.copy(), n_threads=2)
        expected = softmax(a.copy())
        np.testing.assert_array_almost_equal(result, expected)


class TestCumsum:
    def test_basic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = cumsum(a.copy())
        np.testing.assert_array_almost_equal(result, [1.0, 3.0, 6.0, 10.0])

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(1000)
        result = cumsum(a.copy())
        expected = np.cumsum(a)
        np.testing.assert_array_almost_equal(result, expected)


class TestRollingSum:
    def test_basic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_sum(a, 3)
        # Window 3: [1], [1+2], [1+2+3], [2+3+4], [3+4+5]
        np.testing.assert_array_almost_equal(result, [1.0, 3.0, 6.0, 9.0, 12.0])

    def test_window_1(self):
        a = np.array([1.0, 2.0, 3.0])
        result = rolling_sum(a, 1)
        np.testing.assert_array_almost_equal(result, a)


class TestVariance:
    def test_basic(self):
        a = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = variance(a)
        # Mean=5, variance=4
        assert abs(result - 4.0) < 1e-10

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        a = rng.random(1000)
        result = variance(a)
        expected = float(np.var(a))  # Population variance
        assert abs(result - expected) < 1e-10


class TestHistogram:
    def test_basic(self):
        a = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        result = histogram(a, 5, 0.0, 1.0)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0, 1.0, 2.0])

    def test_uniform(self):
        # All values in same bin
        a = np.array([0.5, 0.5, 0.5])
        result = histogram(a, 10, 0.0, 1.0)
        assert result[5] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
