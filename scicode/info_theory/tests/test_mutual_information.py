"""Test cases for mutual information function."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from info_theory.mutual_information import mutual_info


def test_mutual_info():
    # Test Case 1
    channel1 = np.eye(2)
    prior1 = [0.5, 0.5]
    # Noiseless, uniform binary. I = 1 bit.
    target1 = 1.0
    print(f"Test Case 1: Noiseless Channel. Result: {mutual_info(channel1, prior1)} (Expected: {target1})")
    assert np.allclose(mutual_info(channel1, prior1), target1)

    # Test Case 2
    channel2 = np.array([[1/2, 1/2], [1/2, 1/2]])
    prior2 = [3/8, 5/8]
    # Output independent of input. I = 0.
    target2 = 0.0
    print(f"Test Case 2: Random Channel. Result: {mutual_info(channel2, prior2)} (Expected: {target2})")
    assert np.allclose(mutual_info(channel2, prior2), target2)

    # Test Case 3
    channel3 = np.array([[0.8, 0], [0, 0.8], [0.2, 0.2]])
    prior3 = [1/2, 1/2]
    # Manual Calc from thought process: 0.8 bits
    target3 = 0.8
    print(f"Test Case 3: Erasure-like Channel. Result: {mutual_info(channel3, prior3)} (Expected: {target3})")
    assert np.allclose(mutual_info(channel3, prior3), target3)

    print("All tests passed for Mutual Information!")


if __name__ == "__main__":
    test_mutual_info()
