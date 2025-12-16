"""
VQE Submission Module

Contains implementations for VQE (Variational Quantum Eigensolver) subproblems.
"""

import numpy as np


def measureZ(U, psi):
    """
    Measure expectation value of Z⊗I after applying unitary U.

    Args:
        U: 4x4 unitary matrix
        psi: 4x1 quantum state vector

    Returns:
        Expectation value <psi|U†(Z⊗I)U|psi>
    """
    # Apply unitary to state
    phi = U @ psi

    # Z⊗I operator (Z on first qubit, Identity on second)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    Z_I = np.kron(Z, I)

    # Compute expectation value
    expectation = np.conj(phi).T @ Z_I @ phi
    return np.real(expectation).item()
