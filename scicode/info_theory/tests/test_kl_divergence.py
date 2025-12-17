"""Test cases for KL divergence function."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from info_theory.kl_divergence import KL_divergence


def test_kl_divergence():
    # Test Case 1
    p1 = [1/2, 1/2, 0, 0]
    q1 = [1/4, 1/4, 1/4, 1/4]
    # Manual Calc: 0.5*log2(0.5/0.25) + 0.5*log2(0.5/0.25) = 0.5*1 + 0.5*1 = 1.0
    target1 = 1.0
    print(f"Test Case 1: KL({p1}, {q1}) = {KL_divergence(p1, q1)} (Expected: {target1})")
    assert np.allclose(KL_divergence(p1, q1), target1)

    # Test Case 2
    p2 = [1/2, 1/2]
    q2 = [1/2, 1/2]
    # Manual Calc: Identical distributions -> 0
    target2 = 0.0
    print(f"Test Case 2: KL({p2}, {q2}) = {KL_divergence(p2, q2)} (Expected: {target2})")
    assert np.allclose(KL_divergence(p2, q2), target2)

    # Test Case 3
    p3 = [1, 0]
    q3 = [1/4, 3/4]
    # Manual Calc: 1*log2(1/0.25) = log2(4) = 2.0
    target3 = 2.0
    print(f"Test Case 3: KL({p3}, {q3}) = {KL_divergence(p3, q3)} (Expected: {target3})")
    assert np.allclose(KL_divergence(p3, q3), target3)

    print("All tests passed for KL Divergence!")


if __name__ == "__main__":
    test_kl_divergence()
