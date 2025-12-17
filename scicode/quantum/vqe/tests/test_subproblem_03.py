"""
VQE (Problem 59) - Subproblem 59.3

Test cases for:
  measureZ(U, psi)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_vqe_subproblem_03.py
"""

from __future__ import annotations

import importlib
import os

import numpy as np


def _load_fn(fn_name: str):
    module_name = os.environ.get("SCICODE_SUBMISSION_MODULE", "submission")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Cannot import submission module {module_name!r}. "
            "Set SCICODE_SUBMISSION_MODULE to the module containing your solution "
            f"(must define `{fn_name}`)."
        ) from exc
    try:
        return getattr(module, fn_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Submission module {module_name!r} does not define `{fn_name}`."
        ) from exc


def run_tests() -> None:
    measureZ = _load_fn("measureZ")

    CNOT21 = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=complex,
    )

    # Test case 1
    U = CNOT21
    psi = np.kron([[0], [1]], [[0], [1]])
    target = 1.0
    assert np.allclose(measureZ(U, psi), target)

    # Test case 2
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    U = np.dot(CNOT21, np.kron(H, H))
    psi = np.kron([[1], [-1]], [[1], [-1]]) / 2
    target = 1.0
    assert np.allclose(measureZ(U, psi), target)

    # Test case 3
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    U = np.dot(CNOT21, np.kron(np.dot(H, S.conj().T), np.dot(H, S.conj().T)))
    psi = np.kron([[1], [-1j]], [[1], [-1j]]) / 2
    target = 1.0
    assert np.allclose(measureZ(U, psi), target)

    print("✓ Subproblem 59.3: all tests passed")


if __name__ == "__main__":
    run_tests()

