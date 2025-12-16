"""
GACD_rev_coherent_info (Problem 71) - Subproblem 7

Test cases for:
  generalized_amplitude_damping_channel(gamma, N)

Usage:
  SCICODE_SUBMISSION_MODULE=your_module python3 scicode-python/test_GACD_rev_coherent_info_subproblem_07.py
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

def ref_gadc(gamma, N):
    # Formulas:
    # K1 = sqrt(1-N) * (|0><0| + sqrt(1-g)|1><1|)
    # K2 = sqrt(g(1-N)) * |0><1|
    # K3 = sqrt(N) * (sqrt(1-g)|0><0| + |1><1|)
    # K4 = sqrt(gN) * |1><0|
    
    k00 = np.array([[1, 0], [0, 0]])
    k11 = np.array([[0, 0], [0, 1]])
    k01 = np.array([[0, 1], [0, 0]])
    k10 = np.array([[0, 0], [1, 0]])
    
    K1 = np.sqrt(1-N) * (k00 + np.sqrt(1-gamma) * k11)
    K2 = np.sqrt(gamma*(1-N)) * k01
    K3 = np.sqrt(N) * (np.sqrt(1-gamma) * k00 + k11)
    K4 = np.sqrt(gamma*N) * k10
    
    return [K1, K2, K3, K4]

def run_tests() -> None:
    gadc = _load_fn("generalized_amplitude_damping_channel")

    test_args = [(0, 0), (0.8, 0), (0.5, 0.5)]

    for g, n in test_args:
        target_list = ref_gadc(g, n)
        result_list = gadc(g, n)
        
        assert len(result_list) == 4
        for res, tgt in zip(result_list, target_list):
            assert np.allclose(res, tgt)

    print("✓ Subproblem 71.7: all tests passed")

if __name__ == "__main__":
    run_tests()

