"""Mutual Information implementation."""

import numpy as np


def mutual_info(channel, prior):
    """Calculate the mutual information I(X;Y) for a discrete channel.

    Args:
        channel: Channel matrix where Channel[i][j] = P(Y=i | X=j)
                 Shape is (n_outputs, n_inputs)
        prior: Prior distribution over inputs P(X), shape (n_inputs,)

    Returns:
        Mutual information in bits (base 2)
    """
    channel = np.asarray(channel)
    prior = np.asarray(prior)

    # Joint distribution p(x, y) = p(y|x) * p(x)
    # P(Y=i, X=j) = Channel[i][j] * Prior[j]

    # joint[i, j] = channel[i, j] * prior[j]
    joint = channel * prior[np.newaxis, :]

    # Marginal p(y) = sum_x p(x, y) = sum_j joint[i, j] (sum over inputs)
    py = np.sum(joint, axis=1)

    # Marginal p(x) = prior
    px = prior

    # Mutual Info I(X;Y) = sum_x sum_y p(x,y) log2( p(x,y) / (p(x)p(y)) )
    # Handle zeros: if p(x,y) is 0, term is 0.

    mi = 0.0
    rows, cols = joint.shape
    for i in range(rows):  # y
        for j in range(cols):  # x
            pxy = joint[i, j]
            if pxy > 1e-12:
                # p(x) = prior[j]
                # p(y) = py[i]
                mi += pxy * np.log2(pxy / (prior[j] * py[i]))

    return mi
