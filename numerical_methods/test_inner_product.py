"""Test cases for inner_product function (Subproblem 02).

Tests the inner_product function from the Gram-Schmidt orthogonalization task.
The function should compute the Euclidean inner product (dot product) of two vectors.

Reference: note/gram-schmidt-subproblem-02.md
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_science.gram_schmidt import inner_product


class TestInnerProductBasic:
    """Basic test cases from the problem specification."""

    def test_perpendicular_like_vectors(self):
        """Test case 1: u=[3,4], v=[4,3] -> 3*4 + 4*3 = 24."""
        u = np.array([3, 4])
        v = np.array([4, 3])
        result = inner_product(u, v)
        expected = 24.0
        assert np.isclose(result, expected)

    def test_same_vectors(self):
        """Test case 2: u=[3,4], v=[3,4] -> 9 + 16 = 25."""
        u = np.array([3, 4])
        v = np.array([3, 4])
        result = inner_product(u, v)
        expected = 25.0
        assert np.isclose(result, expected)

    def test_4d_vectors(self):
        """Test case 3: 4D vectors."""
        u = np.array([3, 4, 7, 6])
        v = np.array([4, 3, 2, 8])
        result = inner_product(u, v)
        # Expected: 3*4 + 4*3 + 7*2 + 6*8 = 12 + 12 + 14 + 48 = 86
        expected = 86.0
        assert np.isclose(result, expected)


class TestInnerProductProperties:
    """Mathematical properties of inner product."""

    def test_commutative(self):
        """Inner product should be commutative: <u,v> = <v,u>."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        assert np.isclose(inner_product(u, v), inner_product(v, u))

    def test_linearity_scalar(self):
        """Inner product should be linear in scalar multiplication."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        alpha = 2.5
        # <alpha*u, v> = alpha * <u, v>
        assert np.isclose(inner_product(alpha * u, v), alpha * inner_product(u, v))

    def test_linearity_addition(self):
        """Inner product should be linear in addition."""
        u = np.array([1, 2, 3])
        v = np.array([4, 5, 6])
        w = np.array([7, 8, 9])
        # <u + v, w> = <u, w> + <v, w>
        lhs = inner_product(u + v, w)
        rhs = inner_product(u, w) + inner_product(v, w)
        assert np.isclose(lhs, rhs)

    def test_positive_definite(self):
        """<v, v> >= 0 and <v, v> = 0 iff v = 0."""
        # Non-zero vector
        v = np.array([1, 2, 3])
        assert inner_product(v, v) > 0

        # Zero vector
        v_zero = np.array([0, 0, 0])
        assert np.isclose(inner_product(v_zero, v_zero), 0.0)

    def test_norm_relation(self):
        """<v, v> = ||v||^2."""
        v = np.array([3, 4])
        expected_norm_squared = 25.0  # 3^2 + 4^2
        assert np.isclose(inner_product(v, v), expected_norm_squared)


class TestInnerProductOrthogonality:
    """Tests related to orthogonality."""

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have zero inner product."""
        u = np.array([1, 0])
        v = np.array([0, 1])
        assert np.isclose(inner_product(u, v), 0.0)

    def test_orthogonal_3d(self):
        """Orthogonal 3D vectors."""
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
        assert np.isclose(inner_product(u, v), 0.0)

    def test_nearly_orthogonal(self):
        """Nearly orthogonal vectors should have small inner product."""
        u = np.array([1, 0.001])
        v = np.array([0.001, 1])
        result = inner_product(u, v)
        # Should be small: 0.001 + 0.001 = 0.002
        assert np.isclose(result, 0.002)


class TestInnerProductEdgeCases:
    """Edge cases and special inputs."""

    def test_zero_vector(self):
        """Inner product with zero vector should be zero."""
        u = np.array([1, 2, 3])
        v = np.array([0, 0, 0])
        assert np.isclose(inner_product(u, v), 0.0)

    def test_negative_values(self):
        """Inner product with negative values."""
        u = np.array([-1, -2])
        v = np.array([3, 4])
        # Expected: -1*3 + -2*4 = -3 - 8 = -11
        assert np.isclose(inner_product(u, v), -11.0)

    def test_single_element(self):
        """Single element vectors."""
        u = np.array([5])
        v = np.array([3])
        assert np.isclose(inner_product(u, v), 15.0)

    def test_large_vectors(self):
        """Large vectors."""
        n = 1000
        u = np.ones(n)
        v = np.ones(n)
        assert np.isclose(inner_product(u, v), n)

    def test_float_precision(self):
        """Test with floating point values."""
        u = np.array([0.1, 0.2, 0.3])
        v = np.array([0.4, 0.5, 0.6])
        # Expected: 0.04 + 0.10 + 0.18 = 0.32
        assert np.isclose(inner_product(u, v), 0.32)


class TestInnerProductErrors:
    """Error handling tests."""

    def test_mismatched_sizes_raises_error(self):
        """Vectors of different sizes should raise ValueError."""
        u = np.array([1, 2, 3])
        v = np.array([1, 2])
        with pytest.raises(ValueError):
            inner_product(u, v)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
