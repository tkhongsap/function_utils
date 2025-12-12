import numpy as np

from gen_ai_utils.data_science.gram_schmidt import inner_product, normalize, orthogonalize


def test_normalize_vector_1d():
    v = np.array([3, 4])
    out = normalize(v)
    assert out.shape == v.shape
    assert np.allclose(out, np.array([0.6, 0.8]))
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_normalize_preserves_shape_nd():
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2)
    out = normalize(v)
    assert out.shape == v.shape
    assert np.allclose(out, v / np.linalg.norm(v))


def test_normalize_zero_vector_returns_input():
    v = np.zeros((3, 2, 2))
    out = normalize(v)
    assert out.shape == v.shape
    assert np.array_equal(out, v)


def test_inner_product_matches_dot_for_1d():
    u = np.array([3, 4])
    v = np.array([4, 3])
    assert np.isclose(inner_product(u, v), np.dot(u, v))


def test_inner_product_column_vectors():
    u = np.array([3, 4]).reshape(2, 1)
    v = np.array([4, 3]).reshape(2, 1)
    assert np.isclose(inner_product(u, v), 24.0)


def _assert_orthonormal_columns(Q: np.ndarray, atol: float = 1e-8):
    n = Q.shape[1]
    assert np.allclose(Q.T @ Q, np.eye(n), atol=atol)


def _assert_reconstruction_upper_triangular(Q: np.ndarray, A: np.ndarray, atol: float = 1e-8):
    R = Q.T @ A
    assert np.allclose(A, Q @ R, atol=atol)
    assert np.allclose(np.tril(R, k=-1), 0.0, atol=atol)


def test_orthogonalize_3x3_prompt_matrix():
    A = np.array([[0, 1, 1], [1, 0, -1], [1, -1, 0]], dtype=float).T
    Q = orthogonalize(A)

    assert Q.shape == A.shape
    _assert_orthonormal_columns(Q)
    _assert_reconstruction_upper_triangular(Q, A)


def test_orthogonalize_4x4_prompt_matrix():
    A = np.array(
        [[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, -1, 0]], dtype=float
    ).T
    Q = orthogonalize(A)

    assert Q.shape == A.shape
    _assert_orthonormal_columns(Q)
    _assert_reconstruction_upper_triangular(Q, A)

    # Mirrors the subproblem’s example orthogonality check
    assert np.isclose(float(Q[:, 0].T @ Q[:, 3]), 0.0, atol=1e-8)


def test_orthogonalize_5x5_prompt_matrix():
    A = np.array(
        [
            [0, 1, 1, -1, 1],
            [1, 0, -1, 1, -1],
            [1, -1, 0, -1, -1],
            [-1, 1, -1, 0, -1],
            [-1, -1, -1, -1, 0],
        ],
        dtype=float,
    ).T
    Q = orthogonalize(A)

    assert Q.shape == A.shape
    _assert_orthonormal_columns(Q, atol=1e-7)
    _assert_reconstruction_upper_triangular(Q, A, atol=1e-7)
