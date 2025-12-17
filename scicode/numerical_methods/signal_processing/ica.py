"""ICA - Independent Component Analysis (SciCode Task 31)."""

import numpy as np
import numpy.linalg as la
from scipy import signal


def center(X, divide_sd=True):
    """Standardize matrix X along rows.

    Args:
        X: Input matrix with samples as columns
        divide_sd: If True, divide by standard deviation (standardize)

    Returns:
        Centered (and optionally standardized) matrix
    """
    D = X - np.mean(X, axis=1, keepdims=True)
    if divide_sd:
        D = D / np.std(X, axis=1, keepdims=True)
    return D


def whiten(X):
    """Whiten matrix X so that covariance equals identity.

    Args:
        X: Input matrix with samples as columns

    Returns:
        Whitened matrix Z where cov(Z) = I
    """
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    cov = np.cov(X_centered)
    eigenvalues, eigenvectors = la.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    whitening_matrix = D_inv_sqrt @ eigenvectors.T
    Z = whitening_matrix @ X_centered
    return Z


def ica(X, cycles, tol):
    """Perform ICA using FastICA algorithm.

    Args:
        X: Input mixed signals matrix (n_components x n_samples)
        cycles: Maximum number of iterations per component
        tol: Convergence tolerance

    Returns:
        Estimated independent source signals
    """
    X_whitened = whiten(X)
    n_components, n_samples = X_whitened.shape
    W = np.zeros((n_components, n_components))

    for i in range(n_components):
        w = np.random.randn(n_components)
        w = w / la.norm(w)

        for _ in range(cycles):
            w_old = w.copy()
            wx = w @ X_whitened
            g_wx = np.tanh(wx)
            dg_wx = 1 - g_wx ** 2
            w = np.mean(X_whitened * g_wx, axis=1) - np.mean(dg_wx) * w

            for j in range(i):
                w = w - np.dot(w, W[j]) * W[j]

            w = w / la.norm(w)

            if la.norm(w - w_old) < tol:
                break

        W[i] = w

    S_hat = W @ X_whitened
    return S_hat


def create_signals(N=2000):
    """Create test signals for ICA.

    Args:
        N: Number of samples

    Returns:
        X: Mixed signals
        S: Original source signals
    """
    time = np.linspace(0, 8, N)
    s1 = np.sin(2 * time)
    s2 = 2 * np.sign(np.sin(3 * time))
    s3 = 4 * signal.sawtooth(2 * np.pi * time)
    S = np.array([s1, s2, s3])
    A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
    X = A @ S
    return X, S
