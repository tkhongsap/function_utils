"""
GACD_rev_coherent_info (Problem 71) - Subproblem 4

Test cases for:
  syspermute(X, perm, dim)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_GACD_rev_coherent_info_subproblem_04.py
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

def ref_syspermute(X, perm, dim):
    # perm is assumed to be 1-based list of new order, e.g., [2, 1] means system 2 becomes 1st, system 1 becomes 2nd.
    # Actually, standard permutation vector `p` usually means: the i-th axis of new array comes from p[i]-th axis of old array.
    # Let's verify with standard tensor behavior.
    # If we have systems A (dim da), B (dim db). tensor is A x B.
    # We want B x A.
    # The axes of X are (row_A, row_B, col_A, col_B) if we view it as tensor?
    # No, X is usually flattened (da*db, da*db).
    
    # Reshape X to (d1, d2, ..., dn, d1, d2, ..., dn)
    # The axes 0 to n-1 are row indices, n to 2n-1 are col indices.
    
    n = len(dim)
    tensor_shape = tuple(dim) + tuple(dim)
    X_tensor = X.reshape(tensor_shape)
    
    # Adjust perm to 0-based
    perm0 = [p - 1 for p in perm]
    
    # Construct transpose order
    # New row axes: perm0
    # New col axes: [p + n for p in perm0]
    
    transpose_order = perm0 + [p + n for p in perm0]
    
    X_permuted = np.transpose(X_tensor, axes=transpose_order)
    
    # Reshape back to 2D
    # The new dimensions are permuted dimensions
    new_dim = [dim[p] for p in perm0]
    total_dim = np.prod(new_dim)
    
    return X_permuted.reshape(total_dim, total_dim)

def run_tests() -> None:
    syspermute = _load_fn("syspermute")

    # Test 1
    # |0><0| tensor |1><1| = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]] ... wait
    # A = |0><0| = [[1,0],[0,0]]
    # B = |1><1| = [[0,0],[0,1]]
    # X = kron(A, B).
    # perm=[2,1]. Should be kron(B, A).
    A = np.array([[1,0],[0,0]])
    B = np.array([[0,0],[0,1]])
    X1 = np.kron(A, B)
    target1 = ref_syspermute(X1, [2, 1], [2, 2])
    assert np.allclose(syspermute(X1, [2, 1], [2, 2]), target1)

    # Test 2
    # Bell state |Phi+> = (|00> + |11>)/sqrt(2)
    # rho = (|00><00| + |00><11| + |11><00| + |11><11|)/2
    # Permuting 1 and 2 for Bell state should leave it invariant?
    # |00> -> |00>, |11> -> |11>. Yes.
    X2 = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    # Note: test case in prompt uses unnormalized X2? "X = np.array(...)". 
    # Prompt says "X = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])" (unnormalized).
    X2_raw = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
    target2 = ref_syspermute(X2_raw, [2, 1], [2, 2])
    assert np.allclose(syspermute(X2_raw, [2, 1], [2, 2]), target2)

    # Test 3
    # X = kron(Bell, A). 3 systems. dim=[2,2,2].
    # perm=[1,3,2].
    Bell = X2_raw
    A = np.array([[1,0],[0,0]])
    X3 = np.kron(Bell, A)
    # Original: 1, 2, 3.
    # Permuted: 1, 3, 2.
    # System 1 stays 1. System 2 becomes 3. System 3 becomes 2.
    # New state corresponds to kron(system1, system3, system2).
    # Since Bell is on 1&2, and A is on 3.
    # Bell state entanglement is between 1 and 2.
    # In new state, entanglement is between 1 and 3.
    target3 = ref_syspermute(X3, [1, 3, 2], [2, 2, 2])
    assert np.allclose(syspermute(X3, [1, 3, 2], [2, 2, 2]), target3)

    print("✓ Subproblem 71.4: all tests passed")

if __name__ == "__main__":
    run_tests()

