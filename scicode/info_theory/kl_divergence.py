"""KL Divergence implementation."""

import numpy as np


def KL_divergence(p, q):
    """Calculate the Kullback-Leibler divergence between two probability distributions.

    KL(p||q) = sum p[i] * log2(p[i]/q[i])

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        The KL divergence in bits (base 2)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Filter out 0s in p to avoid 0*log(0) (which is 0)
    mask = p > 0

    # Note: The prompt asks to assume same support, but Test Case 1 violates it.
    # In a robust implementation:
    # If p[i] > 0 and q[i] == 0, KL is inf.
    # If p[i] == 0, contribution is 0 regardless of q[i].

    # Calculate KL using base 2
    # KL(p||q) = sum p[i] * log2(p[i]/q[i])

    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
