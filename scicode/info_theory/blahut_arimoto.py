"""Blahut-Arimoto algorithm for computing channel capacity."""

import numpy as np


def blahut_arimoto(channel, e):
    """Compute channel capacity using the Blahut-Arimoto algorithm.

    Args:
        channel: Channel matrix where Channel[i][j] = P(Y=i | X=j)
                 Shape is (n_outputs, n_inputs)
        e: Convergence threshold for rate change between iterations

    Returns:
        Channel capacity in bits (base 2)
    """
    channel = np.asarray(channel)
    n_outputs, n_inputs = channel.shape

    # Initialize prior uniformly
    p = np.ones(n_inputs) / n_inputs

    # Max iterations to prevent infinite loop if e is too small or logic fails
    max_iter = 10000

    rate_old = 0.0

    for _ in range(max_iter):
        # Compute q(y) output distribution based on current p(x)
        # q(y) = sum_j p(j) C(i|j)  (where C[i][j] = p(i|j))
        q = channel @ p

        # Calculate D_j = KL( C[:,j] || q[:] ) for each input j
        # D_j = sum_i C[i,j] * log( C[i,j] / q[i] )

        D = np.zeros(n_inputs)
        for j in range(n_inputs):
            # Column j of channel
            col = channel[:, j]
            # KL divergence
            # Mask zeros
            mask = (col > 0) & (q > 0)
            if np.any(mask):
                D[j] = np.sum(col[mask] * np.log(col[mask] / q[mask]))

        # Update p
        # p_new(j) = p(j) * exp(D[j]) / sum(...)
        numerator = p * np.exp(D)
        p_new = numerator / np.sum(numerator)

        # Calculate Capacity (Mutual Information) in Bits
        # I(X;Y) = sum_j p(j) D_j(bits)
        # But D calculated above is in nats.
        # I_nats = sum(p * D)
        # I_bits = I_nats / log(2)

        rate_new = np.sum(p_new * D) / np.log(2)

        if np.abs(rate_new - rate_old) < e:
            return rate_new

        rate_old = rate_new
        p = p_new

    return rate_new
