"""Test cases for orthogonalize function (Subproblem 03).

Tests the orthogonalize function from the Gram-Schmidt orthogonalization task.
The function should perform Gram-Schmidt orthogonalization on N linearly
independent vectors to produce N orthonormal vectors.

Reference: note/gram-schmidt-subproblem-03.md
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from numerical_methods.linear_algebra.gram_schmidt import orthogonalize, inner_product


class TestOrthogonalizeBasic:
    """Basic test cases from the problem specification."""

    def test_4x4_orthogonality_check(self):
        """Test case 1: Check that column 0 and column 3 are orthogonal."""
        A = np.array([[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, -1, 0]]).T
        B = orthogonalize(A)
        # Check orthogonality between column 0 and column 3
        assert np.isclose(B[:, 0].T @ B[:, 3], 0.0)

    def test_3x3_matrix(self):
        """Test case 2: 3x3 matrix orthogonalization."""
        A = np.array([[0, 1, 1], [1, 0, -1], [1, -1, 0]]).T
        B = orthogonalize(A)
        # Verify all columns are orthonormal
        self._verify_orthonormal(B)

    def test_4x4_matrix(self):
        """Test case 3: 4x4 matrix orthogonalization."""
        A = np.array([[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, -1, 0]]).T
        B = orthogonalize(A)
        # Verify all columns are orthonormal
        self._verify_orthonormal(B)

    def test_5x5_matrix(self):
        """Test case 4: 5x5 matrix orthogonalization."""
        A = np.array([
            [0, 1, 1, -1, 1],
            [1, 0, -1, 1, -1],
            [1, -1, 0, -1, -1],
            [-1, 1, -1, 0, -1],
            [-1, -1, -1, -1, 0]
        ]).T
        B = orthogonalize(A)
        # Verify all columns are orthonormal
        self._verify_orthonormal(B)

    def _verify_orthonormal(self, B):
        """Helper to verify that columns of B are orthonormal."""
        n = B.shape[1]
        # Check orthogonality: B.T @ B should be identity
        product = B.T @ B
        assert np.allclose(product, np.eye(n), atol=1e-10)


class TestOrthogonalizeProperties:
    """Property-based tests for Gram-Schmidt orthogonalization."""

    def test_columns_are_unit_vectors(self):
        """Each column of output should have unit norm."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).astype(float)
        B = orthogonalize(A)
        for j in range(B.shape[1]):
            assert np.isclose(np.linalg.norm(B[:, j]), 1.0)

    def test_columns_are_mutually_orthogonal(self):
        """All pairs of columns should be orthogonal."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).astype(float)
        B = orthogonalize(A)
        n = B.shape[1]
        for i in range(n):
            for j in range(i + 1, n):
                assert np.isclose(B[:, i] @ B[:, j], 0.0, atol=1e-10)

    def test_orthonormal_basis(self):
        """B.T @ B should equal identity matrix."""
        np.random.seed(42)
        A = np.random.randn(5, 5)
        B = orthogonalize(A)
        assert np.allclose(B.T @ B, np.eye(5), atol=1e-10)

    def test_preserves_span(self):
        """The span of orthonormalized vectors should equal span of original."""
        # If A has full rank, then span(B) = span(A)
        # We can verify by checking that A can be expressed as B @ R for some R
        A = np.array([[1, 2], [3, 4]]).astype(float)
        B = orthogonalize(A)
        # R = B.T @ A should give us the coefficients
        R = B.T @ A
        # Reconstructing: B @ R should give us A
        reconstructed = B @ R
        assert np.allclose(reconstructed, A)

    def test_determinant_preserved_up_to_sign(self):
        """Orthonormalization preserves absolute value of determinant for orthonormal matrices."""
        A = np.array([[1, 2], [3, 4]]).astype(float)
        B = orthogonalize(A)
        # For orthonormal matrix, |det(B)| = 1
        assert np.isclose(np.abs(np.linalg.det(B)), 1.0)


class TestOrthogonalizeSpecialCases:
    """Special matrix cases."""

    def test_identity_matrix(self):
        """Identity matrix should remain unchanged (already orthonormal)."""
        A = np.eye(3)
        B = orthogonalize(A)
        assert np.allclose(B, np.eye(3))

    def test_already_orthogonal(self):
        """Already orthogonal matrix should produce same (normalized) result."""
        # Orthogonal but not normalized
        A = np.array([[2, 0], [0, 3]]).astype(float)
        B = orthogonalize(A)
        expected = np.array([[1, 0], [0, 1]]).astype(float)
        assert np.allclose(B, expected)

    def test_2x2_simple(self):
        """Simple 2x2 case."""
        A = np.array([[1, 1], [0, 1]]).astype(float)
        B = orthogonalize(A)
        # First column: [1, 0] normalized = [1, 0]
        # Second column: [1, 1] - proj onto [1, 0] = [1, 1] - [1, 0] = [0, 1]
        expected = np.array([[1, 0], [0, 1]]).astype(float)
        assert np.allclose(B, expected)

    def test_diagonal_matrix(self):
        """Diagonal matrix orthogonalization."""
        A = np.diag([3, 4, 5]).astype(float)
        B = orthogonalize(A)
        # Should produce identity (each basis vector normalized)
        assert np.allclose(B, np.eye(3))


class TestOrthogonalizeRandomMatrices:
    """Tests with random matrices."""

    def test_random_3x3(self):
        """Random 3x3 matrix."""
        np.random.seed(123)
        A = np.random.randn(3, 3)
        B = orthogonalize(A)
        # Verify orthonormality
        assert np.allclose(B.T @ B, np.eye(3), atol=1e-10)

    def test_random_10x10(self):
        """Random 10x10 matrix."""
        np.random.seed(456)
        A = np.random.randn(10, 10)
        B = orthogonalize(A)
        # Verify orthonormality
        assert np.allclose(B.T @ B, np.eye(10), atol=1e-10)

    def test_multiple_random_matrices(self):
        """Multiple random matrices of varying sizes."""
        np.random.seed(789)
        for n in [2, 3, 5, 7]:
            A = np.random.randn(n, n)
            B = orthogonalize(A)
            # Verify orthonormality
            assert np.allclose(B.T @ B, np.eye(n), atol=1e-10), f"Failed for n={n}"


class TestOrthogonalizeNumericalStability:
    """Numerical stability tests."""

    def test_nearly_parallel_vectors(self):
        """Vectors that are nearly parallel (ill-conditioned)."""
        A = np.array([[1, 1.001], [1, 1.002]]).astype(float)
        B = orthogonalize(A)
        # Should still produce orthonormal result
        assert np.allclose(B.T @ B, np.eye(2), atol=1e-6)

    def test_large_values(self):
        """Matrix with large values."""
        A = np.array([[1e6, 2e6], [3e6, 4e6]]).astype(float)
        B = orthogonalize(A)
        assert np.allclose(B.T @ B, np.eye(2), atol=1e-10)

    def test_small_values(self):
        """Matrix with small values."""
        A = np.array([[1e-6, 2e-6], [3e-6, 4e-6]]).astype(float)
        B = orthogonalize(A)
        assert np.allclose(B.T @ B, np.eye(2), atol=1e-10)


class TestOrthogonalizeErrors:
    """Error handling tests."""

    def test_non_square_matrix_raises_error(self):
        """Non-square matrix should raise ValueError."""
        A = np.array([[1, 2, 3], [4, 5, 6]]).astype(float)
        with pytest.raises(ValueError):
            orthogonalize(A)

    def test_1d_array_raises_error(self):
        """1D array should raise ValueError."""
        A = np.array([1, 2, 3]).astype(float)
        with pytest.raises(ValueError):
            orthogonalize(A)


class TestOrthogonalizeIntegration:
    """Integration tests using all three functions together."""

    def test_gram_schmidt_step_by_step(self):
        """Verify Gram-Schmidt process step by step."""
        A = np.array([[1, 1], [0, 1]]).astype(float)

        # Step 1: Normalize first column
        from data_science.gram_schmidt import normalize
        v1 = A[:, 0]
        e1 = normalize(v1)
        assert np.allclose(e1, [1, 0])

        # Step 2: Orthogonalize second column against first
        v2 = A[:, 1]
        proj = inner_product(e1, v2) * e1  # projection of v2 onto e1
        u2 = v2 - proj
        e2 = normalize(u2)
        assert np.allclose(e2, [0, 1])

        # Step 3: Full orthogonalization should match
        B = orthogonalize(A)
        assert np.allclose(B[:, 0], e1)
        assert np.allclose(B[:, 1], e2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
