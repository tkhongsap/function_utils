"""Gram–Schmidt orthogonalization utilities.

Implements subproblems for SciCode task `Gram_Schmidt_orthogonalization`:
- normalize(v)
- inner_product(u, v)
- orthogonalize(A)

Assumes standard Euclidean inner product and Frobenius/Euclidean norm via NumPy.
"""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize the input vector/array.

    Input:
        v (numpy array): The input vector.

    Output:
        n (numpy array): The normalized vector (same shape as v).

    Notes:
        If the norm is 0, this returns v unchanged.
    """
    v_arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v_arr)
    n = v_arr / norm if norm else v_arr
    return n


def inner_product(u: np.ndarray, v: np.ndarray):
    """Calculate the inner product of two vectors.

    Input:
        u (numpy array): Vector 1.
        v (numpy array): Vector 2.

    Output:
        p (float): Inner product of the vectors.

    Notes:
        This treats the inputs as vectors by flattening them.
    """
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)

    if u_arr.size != v_arr.size:
        raise ValueError("u and v must have the same number of elements")

    p = float(np.dot(u_arr.ravel(), v_arr.ravel()))
    return p


def orthogonalize(A: np.ndarray) -> np.ndarray:
    """Perform Gram-Schmidt orthogonalization to produce orthonormal vectors.

    Input:
        A (N*N numpy array): N linearly independent vectors in the N-dimension space.
                            Vectors are assumed to be stored as columns.

    Output:
        B (N*N numpy array): The collection of the orthonormal vectors (columns).
    """
    A_arr = np.asarray(A, dtype=float)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be a square 2D array of shape (N, N)")

    n = A_arr.shape[1]
    B = np.zeros_like(A_arr, dtype=float)

    for j in range(n):
        v = A_arr[:, j].copy()
        for i in range(j):
            # Since B[:, i] is normalized, projection coefficient is <B_i, v>
            v = v - B[:, i] * inner_product(B[:, i], v)
        B[:, j] = normalize(v)

    return B
