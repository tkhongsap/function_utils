"""
GACD_rev_coherent_info (Problem 71) - Subproblem 5

Test cases for:
  partial_trace(X, sys, dim)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_GACD_rev_coherent_info_subproblem_05.py
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

def ref_partial_trace(X, sys, dim):
    # sys is list of systems to trace out (1-based)
    n = len(dim)
    tensor_shape = tuple(dim) + tuple(dim)
    X_tensor = X.reshape(tensor_shape)
    
    sys0 = [s - 1 for s in sys] # 0-based indices to trace
    keep0 = [i for i in range(n) if i not in sys0]
    
    # Move traced axes to the end for both rows and cols?
    # Or use einsum.
    # Einsum string:
    # Indices 0..n-1 (rows), n..2n-1 (cols).
    # We want to contract row_i with col_i for i in sys0.
    # And keep row_j, col_j for j in keep0.
    
    # Create index labels
    # Row indices: 0, 1, ..., n-1
    # Col indices: n, n+1, ..., 2n-1
    # For contraction: if k in sys0, row index = col index.
    
    # This is getting complicated to generate string dynamically.
    # Alternative: Move axes to be traced to end, then trace.
    
    # Let's use summation over diagonal.
    # Trace over k: sum_{i} X_{...i...; ...i...}
    
    curr_tensor = X_tensor
    # We need to trace out specific axes.
    # It is easier if we permute axes so that the kept systems are first, and traced systems are last.
    # Order: [keep_row, keep_col, trace_row, trace_col]
    
    # But wait, to trace, row_k and col_k must be the same index.
    # A cleaner way using reshape/einsum:
    
    # Input indices: r0 r1 ... rn-1, c0 c1 ... cn-1
    # Output indices: r_keep..., c_keep...
    # Contraction: r_trace = c_trace
    
    # Generate labels
    # Use simple integers as labels
    input_labels = list(range(2 * n))
    
    # Map for contraction
    # If i is in sys0, then col index (i+n) should be same label as row index (i).
    # We'll replace label (i+n) with label (i).
    # And we want to sum over these labels (implicit in einsum if label appears twice in input? No, standard einsum requires output spec or implicit summation)
    # Actually, diagonal extraction in einsum is 'ii->i'. Trace is 'ii->'.
    
    labels = list(range(2 * n))
    for s in sys0:
        labels[s + n] = labels[s] # Set col index to match row index
        
    # Output labels: kept row indices, kept col indices
    out_labels = []
    for k in keep0:
        out_labels.append(labels[k])
    for k in keep0:
        out_labels.append(labels[k + n])
        
    # Create einsum string or list
    # Use optimized einsum
    res = np.einsum(curr_tensor, labels, out_labels)
    
    # Reshape result to 2D
    new_dim = [dim[k] for k in keep0]
    total_dim = np.prod(new_dim) if new_dim else 1
    return res.reshape(total_dim, total_dim)


def run_tests() -> None:
    partial_trace = _load_fn("partial_trace")

    # Test 1: Bell state, trace out 2. Should get I/2.
    X1 = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]) # Unnormalized in prompt
    # Trace(2) of |Phi+><Phi+| * 2 = 2 * (I/2) = I.
    # |Phi+> = (|00>+|11>)
    # rho = |00><00| + |00><11| + |11><00| + |11><11|
    # Trace_2:
    # <0|_2 rho |0>_2 + <1|_2 rho |1>_2
    # <0|_2 |00><00| |0>_2 = |0><0|
    # <0|_2 |00><11| |0>_2 = 0
    # <1|_2 |11><11| |1>_2 = |1><1|
    # Sum = |0><0| + |1><1| = I.
    target1 = ref_partial_trace(X1, [2], [2, 2])
    assert np.allclose(partial_trace(X1, [2], [2, 2]), target1)

    # Test 2: Kron product
    # X = A (3x3) kron B (2x2)
    # A = |0><0| (on 3 dim)
    # B = |1><1| (on 2 dim)
    A = np.array([[1,0,0],[0,0,0],[0,0,0]])
    B = np.array([[0,0],[0,1]])
    X2 = np.kron(A, B)
    # Trace out 2 (B). Should get A * Tr(B).
    # Tr(B) = 1.
    # Result = A.
    target2 = ref_partial_trace(X2, [2], [3, 2])
    assert np.allclose(partial_trace(X2, [2], [3, 2]), target2)

    # Test 3: Eye/6.
    # dim=[3, 2].
    # X = I_6 / 6.
    # Trace out 1 (3-dim system).
    # Result is sum of diagonal 3x3 blocks?
    # X is diagonal matrix with 1/6.
    # X = I_3 x I_2 / 6 = (I_3/3) x (I_2/2) ? No.
    # I_6 = I_3 x I_2.
    # Trace_1 (trace out first system). Result is Tr(I_3)*I_2 / 6 = 3 * I_2 / 6 = I_2 / 2.
    X3 = np.eye(6)/6
    target3 = ref_partial_trace(X3, [1], [3, 2])
    assert np.allclose(partial_trace(X3, [1], [3, 2]), target3)

    print("✓ Subproblem 71.5: all tests passed")

if __name__ == "__main__":
    run_tests()

