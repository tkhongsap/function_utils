"""
VQE (Problem 59) - Subproblem 59.2

Test cases for:
  create_ansatz(theta)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_vqe_subproblem_02.py
"""

from __future__ import annotations

import importlib
import os

import numpy as np
from scipy.linalg import expm


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


def _rotation_matrices_reference(axis: int, theta: float) -> np.ndarray:
    if axis == 1:
        return np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )
    if axis == 2:
        return np.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )
    if axis == 3:
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=complex,
        )
    raise ValueError("axis must be 1 (x), 2 (y), or 3 (z)")


def run_tests() -> None:
    create_ansatz = _load_fn("create_ansatz")

    test_thetas = [np.pi / 4, np.pi / 8, np.pi / 6]
    target = True

    for theta in test_thetas:
        psi0 = np.kron([[1], [0]], [[1], [0]])
        I = np.array([[1, 0], [0, 1]])
        psi = np.dot(np.kron(I, _rotation_matrices_reference(1, np.pi)), psi0)

        Sx = np.array([[0, 1], [1, 0]])
        Sy = np.array([[0, -1j], [1j, 0]])

        ansatz_o = np.dot(expm(-1j * theta * np.kron(Sy, Sx)), psi)
        ansatz_c = np.asarray(create_ansatz(theta))

        assert (
            np.isclose(
                np.vdot(ansatz_o, ansatz_c),
                np.linalg.norm(ansatz_o) * np.linalg.norm(ansatz_c),
            )
            == target
        )

    print("✓ Subproblem 59.2: all tests passed")


if __name__ == "__main__":
    run_tests()

