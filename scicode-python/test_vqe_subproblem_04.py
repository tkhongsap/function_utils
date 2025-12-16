"""
VQE (Problem 59) - Subproblem 59.4

Test cases for:
  projective_expected(theta, gl)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_vqe_subproblem_04.py
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


def _create_ansatz_reference(theta: float) -> np.ndarray:
    psi0 = np.kron([[1], [0]], [[1], [0]])
    I = np.eye(2)
    psi = np.dot(np.kron(I, _rotation_matrices_reference(1, np.pi)), psi0)

    Sx = np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return np.dot(expm(-1j * theta * np.kron(Sy, Sx)), psi)


def _expected_energy_reference(theta: float, gl: list[float]) -> float:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Sx = np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = np.array([[1, 0], [0, -1]], dtype=complex)

    Ham = (
        gl[0] * np.kron(I, I)
        + gl[1] * np.kron(Sz, I)
        + gl[2] * np.kron(I, Sz)
        + gl[3] * np.kron(Sz, Sz)
        + gl[4] * np.kron(Sy, Sy)
        + gl[5] * np.kron(Sx, Sx)
    )

    psi = _create_ansatz_reference(theta)
    energy = (psi.conj().T @ Ham @ psi)[0, 0]
    return float(np.real_if_close(energy))


def run_tests() -> None:
    projective_expected = _load_fn("projective_expected")

    g0 = -0.4804
    g1 = -0.4347
    g2 = 0.3435
    g3 = 0.5716
    g4 = 0.0910
    g5 = 0.0910
    gl = [g0, g1, g2, g3, g4, g5]

    # Test case 1
    theta = 0
    target = _expected_energy_reference(theta, gl)
    assert np.allclose(projective_expected(theta, gl), target)

    # Test case 2
    theta = np.pi / 6
    target = _expected_energy_reference(theta, gl)
    assert np.allclose(projective_expected(theta, gl), target)

    # Test case 3 (duplicate in the prompt)
    theta = np.pi / 6
    target = _expected_energy_reference(theta, gl)
    assert np.allclose(projective_expected(theta, gl), target)

    print("✓ Subproblem 59.4: all tests passed")


if __name__ == "__main__":
    run_tests()

