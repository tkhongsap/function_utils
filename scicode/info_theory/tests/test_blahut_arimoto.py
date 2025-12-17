"""Test cases for Blahut-Arimoto algorithm."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from info_theory.blahut_arimoto import blahut_arimoto


def test_blahut_arimoto():
    # Test Case 1: Channel 3x3
    np.random.seed(0)
    channel1 = np.array([[1, 0, 1/4], [0, 1, 1/4], [0, 0, 1/2]])
    e = 1e-8
    val1 = blahut_arimoto(channel1, e)
    target1 = 1.08746284
    print(f"Test Case 1 Result: {val1} (Expected: {target1})")
    assert np.allclose(val1, target1, atol=1e-5)

    # Test Case 2: Symmetric Channel
    channel2 = np.array([[0.1, 0.6], [0.9, 0.4]])
    val2 = blahut_arimoto(channel2, e)
    target2 = 0.21505574
    print(f"Test Case 2 Result: {val2} (Expected: {target2})")
    assert np.allclose(val2, target2, atol=1e-5)

    # Test Case 3: Another channel
    channel3 = np.array([[0.8, 0.5], [0.2, 0.5]])
    val3 = blahut_arimoto(channel3, 1e-5)
    target3 = 0.07315545
    print(f"Test Case 3 Result: {val3} (Expected: {target3})")
    assert np.allclose(val3, target3, atol=1e-5)

    # Test Case 4: BSC(0.2)
    # C = 1 - H(0.2) = 1 - 0.7219 = 0.2781
    bsc = np.array([[0.8, 0.2], [0.2, 0.8]])
    val4 = blahut_arimoto(bsc, 1e-8)
    target4 = 1 - (-0.2 * np.log2(0.2) - 0.8 * np.log2(0.8))
    print(f"Test Case 4 (BSC 0.2): {val4} (Expected ~{target4})")
    assert np.allclose(val4, target4, atol=1e-4)

    # Test Case 5: BEC(0.2)
    # C = 1 - epsilon = 0.8
    # BEC channel matrix:
    # Inputs 0, 1. Outputs 0, E, 1.
    # Input 0 -> 0 (0.8), E (0.2), 1 (0)
    # Input 1 -> 0 (0), E (0.2), 1 (0.8)
    # Matrix shape (3, 2)
    # [[0.8, 0], [0, 0.8], [0.2, 0.2]] (Assuming row 2 is Erasure)
    bec = np.array([[0.8, 0], [0, 0.8], [0.2, 0.2]])
    val5 = blahut_arimoto(bec, 1e-8)
    target5 = 0.8
    print(f"Test Case 5 (BEC 0.2): {val5} (Expected {target5})")
    assert np.allclose(val5, target5, atol=1e-4)

    print("All tests passed for Blahut-Arimoto!")


if __name__ == "__main__":
    test_blahut_arimoto()
