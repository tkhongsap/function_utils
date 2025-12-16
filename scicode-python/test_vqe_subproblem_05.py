"""
VQE (Problem 59) - Subproblem 59.5

Test cases for:
  perform_vqe(gl)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_vqe_subproblem_05.py
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


def perform_diag(gl: list[float]) -> float:
    """
    Calculate the ground-state energy with exact diagonalization.
    Input:
      gl = [g0, g1, g2, g3, g4, g5] : array in size 6
    Output:
      energy : float
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Sx = np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = np.array([[1, 0], [0, -1]], dtype=complex)
    Ham = (
        gl[0] * np.kron(I, I)  # g0 * I
        + gl[1] * np.kron(Sz, I)  # g1 * Z0
        + gl[2] * np.kron(I, Sz)  # g2 * Z1
        + gl[3] * np.kron(Sz, Sz)  # g3 * Z0Z1
        + gl[4] * np.kron(Sy, Sy)  # g4 * Y0Y1
        + gl[5] * np.kron(Sx, Sx)  # g5 * X0X1
    )
    return float(np.linalg.eigvalsh(Ham)[0])


def run_tests() -> None:
    perform_vqe = _load_fn("perform_vqe")
    target = True

    gl_list = [
        [-0.4804, -0.4347, 0.3435, 0.5716, 0.0910, 0.0910],
        [-0.4989, -0.3915, 0.3288, 0.5616, 0.0925, 0.0925],
        [-0.5463, -0.2550, 0.2779, 0.5235, 0.0986, 0.0986],
    ]

    for gl in gl_list:
        assert (np.isclose(perform_diag(gl), perform_vqe(gl))) == target

    print("✓ Subproblem 59.5: all tests passed")


if __name__ == "__main__":
    run_tests()

