"""
GACD_rev_coherent_info (Problem 71) - Subproblem 1

Test cases for:
  ket(dim, args)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_GACD_rev_coherent_info_subproblem_01.py
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

def ref_ket(dim, args):
    if isinstance(dim, int):
        dim = [dim]
        args = [args]
    
    # Check if inputs are valid lists
    if len(dim) != len(args):
        raise ValueError("Dimensions and args must have the same length")
        
    vecs = []
    for d, j in zip(dim, args):
        v = np.zeros(d)
        v[j] = 1
        vecs.append(v)
    
    # Compute tensor product
    res = vecs[0]
    for v in vecs[1:]:
        res = np.kron(res, v)
    return res

def run_tests() -> None:
    ket = _load_fn("ket")

    test_cases = [
        (2, 0),
        (2, [1, 1]), # This implies dim=2 repeated? The prompt says "If d is given as an int and j is given as a list... return tensor product of d-dimensional basis vectors"
        ([2, 3], [0, 1])
    ]

    # Interpreting the prompt for "dim is int and args is list":
    # Prompt: "If d is given as an int and j is given as a list [j1...jn], then return ... of d-dimensional basis vectors."
    # So if dim=2, args=[1,1], it means |1>_2 tensor |1>_2.
    
    for d, a in test_cases:
        # Prepare arguments for reference implementation to match logic
        if isinstance(d, int) and isinstance(a, list):
            ref_d = [d] * len(a)
            ref_a = a
        else:
            ref_d = d
            ref_a = a
            
        target = ref_ket(ref_d, ref_a)
        result = ket(d, a)
        
        assert np.allclose(result, target), f"Failed for dim={d}, args={a}"

    print("✓ Subproblem 71.1: all tests passed")

if __name__ == "__main__":
    run_tests()

