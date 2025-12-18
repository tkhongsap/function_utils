"""
GHZ Protocol Fidelity - Test Cases for All Subproblems
Problem 65 - Quantum Information/Computing

This file contains the test cases extracted from each subproblem.
Each section includes the function signature, docstring, and test cases.
"""

import numpy as np
from scipy.linalg import sqrtm
import itertools


# =============================================================================
# SUBPROBLEM 65.1: Tensor Product Function
# =============================================================================

def tensor(*args):
    """Takes the tensor product of an arbitrary number of matrices/vectors.

    Input:
        args: any number of nd arrays of floats, corresponding to input matrices

    Output:
        M: the tensor product (kronecker product) of input matrices, 2d array of floats

    Note: Original function header was `def tensor():` which is incompatible
          with the test cases. Fixed to `def tensor(*args):` to accept variadic args.
    """
    # TODO: Implement tensor product
    # Hint: Use np.kron iteratively
    pass


def test_subproblem_65_1():
    """Test cases for tensor product function."""
    print("=" * 60)
    print("SUBPROBLEM 65.1: Tensor Product Function")
    print("=" * 60)

    # Test case 1: tensor([0,1],[0,1])
    # Expected: Kronecker product of two [0,1] vectors
    target_1 = np.kron([0, 1], [0, 1])  # [0, 0, 0, 1]
    print(f"Test 1: tensor([0,1], [0,1])")
    print(f"  Expected shape: {target_1.shape}")
    print(f"  Expected value: {target_1}")

    # Test case 2: tensor(np.eye(3), np.ones((3,3)))
    target_2 = np.kron(np.eye(3), np.ones((3, 3)))
    print(f"\nTest 2: tensor(np.eye(3), np.ones((3,3)))")
    print(f"  Expected shape: {target_2.shape}")
    print(f"  Expected value:\n{target_2}")

    # Test case 3: tensor([[1/2,1/2],[0,1]], [[1,2],[3,4]])
    target_3 = np.kron([[1/2, 1/2], [0, 1]], [[1, 2], [3, 4]])
    print(f"\nTest 3: tensor([[1/2,1/2],[0,1]], [[1,2],[3,4]])")
    print(f"  Expected shape: {target_3.shape}")
    print(f"  Expected value:\n{target_3}")

    print()


# =============================================================================
# SUBPROBLEM 65.2: Apply Channel Function
# =============================================================================

def apply_channel(K, rho, sys=None, dim=None):
    """Applies the channel with Kraus operators in K to the state rho on
    systems specified by the list sys. The dimensions of the subsystems of
    rho are given by dim.

    Inputs:
        K: list of 2d array of floats, list of Kraus operators
        rho: 2d array of floats, input density matrix
        sys: list of int or None, list of subsystems to apply the channel, None means full system
        dim: list of int or None, list of dimensions of each subsystem, None means full system

    Output:
        matrix: output density matrix of floats

    Note: Test cases use 1-based indexing for sys (e.g., sys=[2] for second qubit
          in a 2-qubit system), but this is NOT documented in the prompt.
    """
    # TODO: Implement channel application
    pass


def test_subproblem_65_2():
    """Test cases for apply_channel function."""
    print("=" * 60)
    print("SUBPROBLEM 65.2: Apply Channel Function")
    print("=" * 60)

    # Test case 1: Identity channel on single qubit
    K_1 = [np.eye(2)]
    rho_1 = np.array([[0.8, 0], [0, 0.2]])
    print("Test 1: Identity channel on single qubit state")
    print(f"  K = [np.eye(2)]")
    print(f"  rho = [[0.8, 0], [0, 0.2]]")
    print(f"  sys=None, dim=None")
    print(f"  Expected: rho unchanged = {rho_1}")

    # Test case 2: Dephasing channel (complete) on single qubit
    K_2 = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]
    rho_2 = np.ones((2, 2)) / 2
    # This channel completely dephases: rho -> diagonal part only
    target_2 = np.array([[0.5, 0], [0, 0.5]])
    print(f"\nTest 2: Complete dephasing channel")
    print(f"  K = [|0><0|, |1><1|]")
    print(f"  rho = maximally mixed state")
    print(f"  Expected: diagonal matrix {np.diag(target_2)}")

    # Test case 3: Partial channel on 2-qubit system (second qubit only)
    K_3 = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]
    rho_3 = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
    print(f"\nTest 3: Dephasing on second qubit of Bell state")
    print(f"  K = [|0><0|, |1><1|]")
    print(f"  rho = |Phi+><Phi+| (Bell state)")
    print(f"  sys=[2], dim=[2,2]  <-- NOTE: Uses 1-based indexing!")
    print(f"  This applies channel to second qubit only")

    print()


# =============================================================================
# SUBPROBLEM 65.3: Channel Output Function
# =============================================================================

def channel_output(input_state, channel1, channel2=None):
    """Returns the channel output.

    Inputs:
        input_state: density matrix of the input 2n qubit state, (2**(2n), 2**(2n)) array of floats
        channel1: Kraus operators of the first channel, list of (2,2) array of floats
        channel2: Kraus operators of the second channel, list of (2,2) array of floats

    Output:
        output: the channel output, (2**(2n), 2**(2n)) array of floats

    Note: If channel2 is None, it defaults to channel1.
    """
    # TODO: Implement using apply_channel
    pass


def test_subproblem_65_3():
    """Test cases for channel_output function."""
    print("=" * 60)
    print("SUBPROBLEM 65.3: Channel Output Function")
    print("=" * 60)

    # Test case 1: Bit-flip channel on Bell state (same channel both qubits)
    input_state_1 = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1]
    ]) / 2
    channel1_1 = [np.array([[0, 1], [1, 0]])]  # X gate (bit flip)
    print("Test 1: Bit-flip (X gate) on Bell state")
    print(f"  input_state = |Phi+><Phi+|")
    print(f"  channel1 = [X] (bit flip)")
    print(f"  channel2 = None (defaults to channel1)")
    # X on both qubits of |00> + |11> gives |11> + |00> = same state

    # Test case 2: Dephasing channel on entangled state
    input_state_2 = np.array([
        [0.8, 0, 0, np.sqrt(0.16)],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [np.sqrt(0.16), 0, 0, 0.2]
    ])
    channel1_2 = [
        np.array([[np.sqrt(0.5), 0], [0, np.sqrt(0.5)]]),
        np.array([[np.sqrt(0.5), 0], [0, -np.sqrt(0.5)]])
    ]
    print(f"\nTest 2: Dephasing channel on partially entangled state")
    print(f"  input_state: partially entangled state")
    print(f"  channel1 = dephasing channel (p=0.5)")

    # Test case 3: Different channels on each qubit
    input_state_3 = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1]
    ]) / 2
    channel1_3 = [np.array([[0, 1], [1, 0]])]  # X gate
    channel2_3 = [np.eye(2)]  # Identity
    print(f"\nTest 3: Different channels on each qubit")
    print(f"  input_state = |Phi+><Phi+|")
    print(f"  channel1 = [X] (bit flip on first qubit)")
    print(f"  channel2 = [I] (identity on second qubit)")

    print()


# =============================================================================
# SUBPROBLEM 65.4: GHZ Protocol
# =============================================================================

def ghz_protocol(state):
    """Returns the output state of the protocol.

    Input:
        state: 2n qubit input state, 2^2n by 2^2n array of floats

    Output:
        post_selected: the output state (2x2 density matrix after post-selection)

    Protocol:
        1. Parity measurement on first n qubits
        2. Parity measurement on last n qubits
        3. Keep if both have "even parity" (interpreted as all-0 or all-1)
        4. Transform |0...0> -> |0> and |1...1> -> |1>

    Note: The prompt says "even parity (i.e., all 0 or all 1)" but these are
          mathematically different conditions for n >= 2.
    """
    # TODO: Implement GHZ protocol
    pass


def test_subproblem_65_4():
    """Test cases for ghz_protocol function."""
    print("=" * 60)
    print("SUBPROBLEM 65.4: GHZ Protocol")
    print("=" * 60)

    # Test case 1: Pure state close to GHZ
    state_1 = np.zeros(16)
    state_1[0] = np.sqrt(0.8)   # |0000> component
    state_1[-1] = np.sqrt(0.2)  # |1111> component
    state_1 = np.outer(state_1, state_1)
    print("Test 1: Pure state with |0000> and |1111> components")
    print(f"  state = sqrt(0.8)|0000> + sqrt(0.2)|1111>")
    print(f"  state shape: {state_1.shape}")
    print(f"  Expected output: 2x2 density matrix")

    # Test case 2: Diagonal mixed state
    state_2 = np.diag([0.3, 0.05, 0, 0.05, 0.05, 0, 0, 0.05] * 2)
    print(f"\nTest 2: Diagonal mixed state")
    print(f"  state shape: {state_2.shape}")
    print(f"  Diagonal: {np.diag(state_2)}")

    # Test case 3: Maximally mixed state
    state_3 = np.ones((16, 16)) / 16
    print(f"\nTest 3: Maximally mixed 4-qubit state")
    print(f"  state = I/16")
    print(f"  state shape: {state_3.shape}")

    print()


# =============================================================================
# SUBPROBLEM 65.5: Fidelity Function
# =============================================================================

def fidelity(rho, sigma):
    """Returns the fidelity between two states.

    Inputs:
        rho, sigma: density matrices of the two states, 2d array of floats

    Output:
        fid: fidelity, float

    Note: The prompt doesn't specify which fidelity definition to use:
        - Uhlmann-Jozsa: F = [Tr(sqrt(sqrt(rho) sigma sqrt(rho)))]^2
        - Root fidelity: F = Tr(sqrt(sqrt(rho) sigma sqrt(rho)))
    """
    # TODO: Implement fidelity calculation
    pass


def test_subproblem_65_5():
    """Test cases for fidelity function."""
    print("=" * 60)
    print("SUBPROBLEM 65.5: Fidelity Function")
    print("=" * 60)

    # Test case 1: Bell state vs maximally mixed
    rho_1 = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1]
    ]) / 2
    sigma_1 = np.eye(4) / 4
    print("Test 1: Bell state |Phi+> vs maximally mixed state")
    print(f"  rho = |Phi+><Phi+|")
    print(f"  sigma = I/4")
    # For pure state rho, F(rho, sigma) = <psi|sigma|psi> = Tr(rho * sigma)
    # = Tr(|Phi+><Phi+| * I/4) = 1/4
    # But this is for squared fidelity; root fidelity would be 1/2

    # Test case 2: Pure states (one is |+>)
    rho_2 = np.array([[1/2, 1/2], [1/2, 1/2]])  # |+><+|
    sigma_2 = np.array([[1, 0], [0, 0]])  # |0><0|
    print(f"\nTest 2: |+> state vs |0> state")
    print(f"  rho = |+><+|")
    print(f"  sigma = |0><0|")
    # F = |<+|0>|^2 = |1/sqrt(2)|^2 = 1/2 (squared)
    # or sqrt(1/2) = 0.707 (root)

    # Test case 3: Orthogonal pure states
    rho_3 = np.array([[1/2, 1/2], [1/2, 1/2]])   # |+><+|
    sigma_3 = np.array([[1/2, -1/2], [-1/2, 1/2]])  # |-><-|
    print(f"\nTest 3: |+> state vs |-> state (orthogonal)")
    print(f"  rho = |+><+|")
    print(f"  sigma = |-><-|")
    print(f"  Expected: 0 (orthogonal states)")

    print()


# =============================================================================
# SUBPROBLEM 65.6: GHZ Protocol Fidelity
# =============================================================================

def ghz_protocol_fidelity(input_state, channel1, channel2=None):
    """Returns the achievable fidelity of the protocol.

    Inputs:
        input_state: density matrix of the input 2n qubit state, (2**(2n), 2**(2n)) array of floats
        channel1: Kraus operators of the first channel, list of (2,2) array of floats
        channel2: Kraus operators of the second channel, list of (2,2) array of floats

    Output:
        fid: achievable fidelity of protocol, float

    Note: The prompt says "fidelity with respect to the two-qubit maximally
          entangled state" but doesn't specify which of the 4 Bell states.
    """
    # TODO: Implement using channel_output, ghz_protocol, and fidelity
    pass


def test_subproblem_65_6():
    """Test cases for ghz_protocol_fidelity function."""
    print("=" * 60)
    print("SUBPROBLEM 65.6: GHZ Protocol Fidelity")
    print("=" * 60)

    # Create 4-qubit GHZ state: (|0000> + |1111>)/sqrt(2)
    ghz = np.zeros(16)
    ghz[0] = 1 / np.sqrt(2)   # |0000>
    ghz[-1] = 1 / np.sqrt(2)  # |1111>
    ghz = np.outer(ghz, ghz)

    # Test case 1: Dephasing channel
    dephasing = [
        np.array([[np.sqrt(0.8), 0], [0, np.sqrt(0.8)]]),
        np.array([[np.sqrt(0.2), 0], [0, -np.sqrt(0.2)]])
    ]
    print("Test 1: GHZ state through dephasing channel")
    print(f"  input_state = 4-qubit GHZ state")
    print(f"  channel = dephasing (p=0.2)")

    # Test case 2: Bit-flip channel
    bitflip = [
        np.array([[np.sqrt(0.8), 0], [0, np.sqrt(0.8)]]),
        np.array([[0, np.sqrt(0.2)], [np.sqrt(0.2), 0]])
    ]
    print(f"\nTest 2: GHZ state through bit-flip channel")
    print(f"  channel = bit-flip (p=0.2)")

    # Test case 3: Y-error channel
    y_error = [
        np.array([[np.sqrt(0.9), 0], [0, np.sqrt(0.9)]]),
        np.array([[0, -1j * np.sqrt(0.1)], [1j * np.sqrt(0.1), 0]])
    ]
    print(f"\nTest 3: GHZ state through Y-error channel")
    print(f"  channel = Y-error (p=0.1)")

    # Test case 4: Identity channel (no noise)
    identity = [np.eye(2)]
    print(f"\nTest 4: GHZ state through identity channel (no noise)")
    print(f"  Expected fidelity: 1.0 (perfect)")

    # Test case 5: Complete dephasing
    comp_dephasing = [
        np.array([[np.sqrt(0.5), 0], [0, np.sqrt(0.5)]]),
        np.array([[np.sqrt(0.5), 0], [0, -np.sqrt(0.5)]])
    ]
    print(f"\nTest 5: GHZ state through complete dephasing")
    print(f"  channel = complete dephasing (p=0.5)")

    print()


# =============================================================================
# HELPER: Reference Implementations for Verification
# =============================================================================

def reference_tensor(*args):
    """Reference implementation of tensor product."""
    if len(args) == 0:
        return np.array([[1]])
    result = np.array(args[0])
    for arr in args[1:]:
        result = np.kron(result, arr)
    return result


def reference_fidelity_squared(rho, sigma):
    """Reference implementation of squared fidelity (Nielsen & Chuang convention)."""
    sqrt_rho = sqrtm(rho)
    inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return np.real(np.trace(inner)) ** 2


def reference_fidelity_root(rho, sigma):
    """Reference implementation of root fidelity (Wilde convention)."""
    sqrt_rho = sqrtm(rho)
    inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return np.real(np.trace(inner))


# =============================================================================
# MAIN: Run all tests
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GHZ PROTOCOL FIDELITY - TEST CASES")
    print("Problem 65 - Quantum Information/Computing")
    print("=" * 60 + "\n")

    test_subproblem_65_1()
    test_subproblem_65_2()
    test_subproblem_65_3()
    test_subproblem_65_4()
    test_subproblem_65_5()
    test_subproblem_65_6()

    print("=" * 60)
    print("REFERENCE IMPLEMENTATION EXAMPLES")
    print("=" * 60)

    # Demo reference tensor
    print("\nReference tensor([0,1], [0,1]):")
    print(f"  Result: {reference_tensor([0,1], [0,1])}")

    # Demo reference fidelity
    rho = np.array([[1, 0], [0, 0]])  # |0><0|
    sigma = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+><+|
    print(f"\nReference fidelity(|0><0|, |+><+|):")
    print(f"  Squared (Nielsen & Chuang): {reference_fidelity_squared(rho, sigma):.4f}")
    print(f"  Root (Wilde): {reference_fidelity_root(rho, sigma):.4f}")

    print("\n" + "=" * 60)
    print("END OF TESTS")
    print("=" * 60 + "\n")
