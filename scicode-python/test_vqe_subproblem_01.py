"""
VQE (Problem 59) - Subproblem 59.1

Test cases for:
  rotation_matrices(axis, theta)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_vqe_subproblem_01.py
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
    rotation_matrices = _load_fn("rotation_matrices")

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    test_vectors = [
        (1, np.pi, X),
        (2, np.pi, Y),
        (3, np.pi, Z),
    ]

    for axis, theta, target in test_vectors:
        assert np.allclose(1j * rotation_matrices(axis, theta), target)

    print("✓ Subproblem 59.1: all tests passed")


if __name__ == "__main__":
    run_tests()

