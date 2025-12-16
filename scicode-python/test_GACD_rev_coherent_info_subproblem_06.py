"""
GACD_rev_coherent_info (Problem 71) - Subproblem 6

Test cases for:
  entropy(rho)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_GACD_rev_coherent_info_subproblem_06.py
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

def ref_entropy(rho):
    # Von Neumann entropy S = -Tr(rho log2 rho)
    # sum -lambda log2 lambda
    vals = np.linalg.eigvalsh(rho)
    # remove zeros or negative numbers due to precision
    vals = vals[vals > 1e-15]
    return -np.sum(vals * np.log2(vals))

def run_tests() -> None:
    entropy = _load_fn("entropy")

    # Test 1: Maximally mixed state dim 4
    # rho = I/4. S = log2(4) = 2.
    rho1 = np.eye(4)/4
    target1 = ref_entropy(rho1)
    assert np.allclose(entropy(rho1), target1)

    # Test 2: Maximally mixed state dim 3
    # S = log2(3)
    rho2 = np.ones((3,3))/3 # Wait, np.ones/3 is pure state?
    # np.ones((3,3)) is matrix of all 1s.
    # Eigenvalues are [3, 0, 0].
    # rho = ones/3 -> eigenvalues [1, 0, 0].
    # This is a pure state |+><+|.
    # Entropy should be 0.
    # Check if prompt meant np.eye(3)/3?
    # Prompt: "rho = np.ones((3,3))/3\nassert np.allclose(entropy(rho), target)"
    # A matrix of all 1s divided by 3 has trace 3/3 + 3/3 + 3/3 = 3? No, trace is 1/3+1/3+1/3 = 1.
    # It is a valid density matrix (pure).
    target2 = ref_entropy(rho2)
    assert np.allclose(entropy(rho2), target2)

    # Test 3: mixed diagonal
    rho3 = np.diag([0.8, 0.2])
    target3 = ref_entropy(rho3)
    assert np.allclose(entropy(rho3), target3)

    print("✓ Subproblem 71.6: all tests passed")

if __name__ == "__main__":
    run_tests()

