"""
GACD_rev_coherent_info (Problem 71) - Subproblem 2

Test cases for:
  tensor(*args)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_GACD_rev_coherent_info_subproblem_02.py
"""

from __future__ import annotations

import importlib
import os
import numpy as np
import functools

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

def ref_tensor(*args):
    return functools.reduce(np.kron, args)

def run_tests() -> None:
    tensor = _load_fn("tensor")

    # Test Case 1
    args1 = ([0, 1], [0, 1])
    target1 = ref_tensor(*args1)
    
    # Test Case 2
    args2 = (np.eye(3), np.ones((3, 3)))
    target2 = ref_tensor(*args2)
    
    # Test Case 3
    args3 = (np.array([[0.5, 0.5], [0, 1]]), np.array([[1, 2], [3, 4]]))
    target3 = ref_tensor(*args3)

    assert np.allclose(tensor(*args1), target1)
    assert np.allclose(tensor(*args2), target2)
    assert np.allclose(tensor(*args3), target3)

    print("✓ Subproblem 71.2: all tests passed")

if __name__ == "__main__":
    run_tests()

