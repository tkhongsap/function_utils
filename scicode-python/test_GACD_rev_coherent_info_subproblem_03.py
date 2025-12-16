"""
GACD_rev_coherent_info (Problem 71) - Subproblem 3

Test cases for:
  apply_channel(K, rho, sys=None, dim=None)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_GACD_rev_coherent_info_subproblem_03.py
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

def ref_apply_channel(K, rho, sys=None, dim=None):
    if sys is None:
        # Apply to full system
        # sum K_i rho K_i^\dagger
        new_rho = np.zeros_like(rho, dtype=complex)
        for k in K:
            new_rho += k @ rho @ k.conj().T
        return new_rho
    else:
        # Apply to subsystem
        # Construct full K_i
        # sys is list of indices (1-based likely given previous context, but let's check prompt/examples)
        # 71_71.3 Test Case 3: sys=[2], dim=[2,2]. This implies 1-based indexing (2nd system).
        
        # Adjust to 0-based
        sys_indices = [s - 1 for s in sys]
        
        new_rho = np.zeros_like(rho, dtype=complex)
        
        for k in K:
            # Construct full operator U_i = I x ... x k x ... x I
            # We need to assume k acts on the specified subsystems combined, 
            # OR k is a list corresponding to sys?
            # Prompt: "K: list of 2d array ... list of Kraus operators".
            # "If the quantum channel acts on the i-th subsystem ... Kraus operators have form I...K_i...I"
            # It implies the provided K_i acts on the subsystems specified in `sys`.
            # Typically if sys has multiple indices, K_i acts on that composite space.
            
            # Since dim is given, we can construct the identity matrices.
            # We'll build the list of operators to tensor product.
            
            # But wait, if sys has multiple indices, how do we tensor it?
            # Example: sys=[2], dim=[2,2]. k is 2x2.
            # op_list = [I(2), k] -> kron(I, k).
            
            # What if sys=[1, 3], dim=[2, 2, 2]? 
            # It's tricky to insert a multi-partite K into specific slots if they are not contiguous or if K is given as a single matrix for those slots.
            # However, standard "Kraus operator on subsystem" usually implies single subsystem or contiguous. 
            # Given test case 3: sys=[2], dim=[2,2]. Single target.
            
            # Let's assume K_i acts on the composite Hilbert space of the specified subsystems.
            # But calculating the full matrix with non-contiguous subsystems is hard with just np.kron order.
            # However, usually we can construct it if we permute dimensions, apply, and permute back.
            # OR simpler: if sys contains just one index, it is easy.
            # If the prompt implies general sys, it might be complicated.
            # Test case 3 only shows single subsystem [2].
            
            # Implementation for single subsystem or contiguous:
            ops = []
            current_sys_idx = 0
            
            # Note: This reference implementation is simplified for the test cases provided which seem to target single subsystems or full systems.
            # If more complex cases arose, we'd need a more robust tensor expansion (like permutation).
            
            # For the purpose of these specific test cases:
            # Case 1 & 2: sys=None.
            # Case 3: sys=[2], dim=[2,2].
            
            if len(sys) == 1:
                target_idx = sys[0] - 1
                ops_list = []
                for i, d in enumerate(dim):
                    if i == target_idx:
                        ops_list.append(k)
                    else:
                        ops_list.append(np.eye(d))
                full_k = ref_tensor(*ops_list)
            else:
                 raise NotImplementedError("Reference implementation only supports single subsystem or None for now")
            
            new_rho += full_k @ rho @ full_k.conj().T
            
        return new_rho

def run_tests() -> None:
    apply_channel = _load_fn("apply_channel")

    # Test 1
    K1 = [np.eye(2)]
    rho1 = np.array([[0.8,0],[0,0.2]])
    target1 = ref_apply_channel(K1, rho1, None, None)
    assert np.allclose(apply_channel(K1, rho1, None, None), target1)

    # Test 2
    K2 = [np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]])]
    rho2 = np.ones((2,2))/2
    target2 = ref_apply_channel(K2, rho2, None, None)
    assert np.allclose(apply_channel(K2, rho2, None, None), target2)

    # Test 3
    K3 = [np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]])]
    # rho is 4x4
    rho3 = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    # sys=[2] (2nd system), dim=[2,2]
    target3 = ref_apply_channel(K3, rho3, sys=[2], dim=[2,2])
    assert np.allclose(apply_channel(K3, rho3, sys=[2], dim=[2,2]), target3)

    print("✓ Subproblem 71.3: all tests passed")

if __name__ == "__main__":
    run_tests()

