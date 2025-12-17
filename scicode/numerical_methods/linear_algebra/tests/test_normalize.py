"""Test cases for normalize function (Subproblem 01).

Tests the normalize function from the Gram-Schmidt orthogonalization task.
The function should normalize input vectors to unit length using Euclidean norm.

Reference: note/gram-schmidt-subproblem-01.md
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from numerical_methods.linear_algebra.gram_schmidt import normalize


class TestNormalizeBasic:
    """Basic test cases from the problem specification."""

    def test_2d_vector(self):
        """Test case 1: Simple 2D vector [3, 4] -> unit vector."""
        v = np.array([3, 4])
        result = normalize(v)
        # Expected: [3/5, 4/5] = [0.6, 0.8]
        expected = np.array([0.6, 0.8])
        assert np.allclose(result, expected)
        # Verify unit norm
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_2d_array_4x2(self):
        """Test case 2: 4x2 array normalized as a single entity."""
        v = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2)
        result = normalize(v)
        # Expected: v / ||v|| where ||v|| = sqrt(1+4+9+16+25+36+49+64) = sqrt(204)
        norm = np.sqrt(204)
        expected = v / norm
        assert np.allclose(result, expected)
        # Verify unit Frobenius norm
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_3d_array_3x2x2(self):
        """Test case 3: 3D array (3x2x2) normalized."""
        v = np.array([i for i in range(12)]).reshape(3, 2, 2)
        result = normalize(v)
        # Expected: v / ||v|| where ||v|| = sqrt(sum of squares 0..11)
        # = sqrt(0+1+4+9+16+25+36+49+64+81+100+121) = sqrt(506)
        norm = np.sqrt(506)
        expected = v / norm
        assert np.allclose(result, expected)
        # Verify unit Frobenius norm
        assert np.isclose(np.linalg.norm(result), 1.0)


class TestNormalizeEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_vector(self):
        """Zero vector should return zero vector (no division by zero)."""
        v = np.array([0, 0, 0])
        result = normalize(v)
        expected = np.array([0.0, 0.0, 0.0])
        assert np.allclose(result, expected)

    def test_unit_vector(self):
        """Already normalized vector should remain unchanged."""
        v = np.array([1, 0, 0])
        result = normalize(v)
        assert np.allclose(result, v)
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_negative_values(self):
        """Vector with negative values."""
        v = np.array([-3, 4])
        result = normalize(v)
        expected = np.array([-0.6, 0.8])
        assert np.allclose(result, expected)

    def test_single_element(self):
        """Single element vector."""
        v = np.array([5])
        result = normalize(v)
        expected = np.array([1.0])
        assert np.allclose(result, expected)

    def test_very_small_values(self):
        """Vector with very small values (numerical stability)."""
        v = np.array([1e-10, 1e-10])
        result = normalize(v)
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        assert np.allclose(result, expected)


class TestNormalizeProperties:
    """Property-based tests for normalize function."""

    def test_output_has_unit_norm(self):
        """Normalized non-zero vector should have unit norm."""
        np.random.seed(42)
        for _ in range(10):
            v = np.random.randn(5)
            result = normalize(v)
            assert np.isclose(np.linalg.norm(result), 1.0)

    def test_preserves_direction(self):
        """Normalization should preserve direction (parallel to original)."""
        v = np.array([3, 4, 5])
        result = normalize(v)
        # Check that result is parallel to v (cross product is zero in 3D)
        # Or simply check that v = alpha * result for some alpha > 0
        alpha = v[0] / result[0]
        assert np.allclose(v, alpha * result)

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        shapes = [(3,), (4, 4), (2, 3, 4)]
        for shape in shapes:
            v = np.random.randn(*shape)
            result = normalize(v)
            assert result.shape == v.shape

    def test_idempotent_for_unit_vectors(self):
        """Normalizing a unit vector should return the same vector."""
        v = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
        result = normalize(v)
        assert np.allclose(result, v)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
