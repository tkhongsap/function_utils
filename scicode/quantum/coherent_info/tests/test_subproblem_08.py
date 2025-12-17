"""
GACD_rev_coherent_info (Problem 71) - Subproblem 8

Test cases for:
  neg_rev_coh_info(p, g, N)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_GACD_rev_coherent_info_subproblem_08.py
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

# Helpers from previous steps
def ref_entropy(rho):
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return -np.sum(vals * np.log2(vals))

def ref_gadc(gamma, N):
    k00 = np.array([[1, 0], [0, 0]])
    k11 = np.array([[0, 0], [0, 1]])
    k01 = np.array([[0, 1], [0, 0]])
    k10 = np.array([[0, 0], [1, 0]])
    
    K1 = np.sqrt(1-N) * (k00 + np.sqrt(1-gamma) * k11)
    K2 = np.sqrt(gamma*(1-N)) * k01
    K3 = np.sqrt(N) * (np.sqrt(1-gamma) * k00 + k11)
    K4 = np.sqrt(gamma*N) * k10
    return [K1, K2, K3, K4]

def ref_partial_trace(X, sys, dim):
    n = len(dim)
    tensor_shape = tuple(dim) + tuple(dim)
    X_tensor = X.reshape(tensor_shape)
    sys0 = [s - 1 for s in sys]
    keep0 = [i for i in range(n) if i not in sys0]
    labels = list(range(2 * n))
    for s in sys0:
        labels[s + n] = labels[s]
    out_labels = []
    for k in keep0:
        out_labels.append(labels[k])
    for k in keep0:
        out_labels.append(labels[k + n])
    res = np.einsum(X_tensor, labels, out_labels)
    new_dim = [dim[k] for k in keep0]
    total_dim = np.prod(new_dim) if new_dim else 1
    return res.reshape(total_dim, total_dim)

def ref_neg_rev_coh_info(p, g, N):
    # State: sqrt(1-p)|00> + sqrt(p)|11>
    # psi = vector
    v0 = np.array([1, 0])
    v1 = np.array([0, 1])
    psi = np.sqrt(1-p) * np.kron(v0, v0) + np.sqrt(p) * np.kron(v1, v1)
    rho_in = np.outer(psi, psi)
    
    # Apply GADC to one qubit.
    # "Sending one qubit... through GADC". Usually implies standard setting.
    # Background 71.9 says rho = id_A x A_A'->B (psi_AA')
    # So channel acts on the second qubit.
    
    Ks = ref_gadc(g, N)
    
    # Apply channel to system 2
    # dim = [2, 2]
    # K_full = I x K_i
    
    rho_out = np.zeros_like(rho_in, dtype=complex)
    for k in Ks:
        full_k = np.kron(np.eye(2), k)
        rho_out += full_k @ rho_in @ full_k.conj().T
        
    # Reverse coherent info I_R(A>B) = S(A) - S(AB)
    # A is system 1 (unchanged reference).
    # AB is system 1 + 2 (output).
    
    # S(AB)
    S_AB = ref_entropy(rho_out)
    
    # S(A) -> trace out B (system 2)
    rho_A = ref_partial_trace(rho_out, [2], [2, 2])
    S_A = ref_entropy(rho_A)
    
    I_R = S_A - S_AB
    return -I_R

def run_tests() -> None:
    neg_rev_coh_info = _load_fn("neg_rev_coh_info")

    test_cases = [
        (0.477991, 0.2, 0.4),
        (0.407786, 0.2, 0.1),
        (0.399685, 0.4, 0.2)
    ]

    for p, g, n in test_cases:
        target = ref_neg_rev_coh_info(p, g, n)
        res = neg_rev_coh_info(p, g, n)
        assert np.allclose(res, target)

    print("✓ Subproblem 71.8: all tests passed")

if __name__ == "__main__":
    run_tests()

