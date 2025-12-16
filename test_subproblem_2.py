import numpy as np

def mutual_info(channel, prior):
    # Placeholder for the user's implementation
    # Reference implementation
    channel = np.asarray(channel) # Shape (Ny, Nx) or (Nx, Ny)? Prompt says "Channel[i][j] means probability of i given j"
    # Typically i is output, j is input.
    # But let's verify with test cases in test_mutual_info logic.
    prior = np.asarray(prior)
    
    # Check dimensions
    # prior has shape (Nx,)
    # If channel is (Ny, Nx), then channel @ prior = p(y)
    
    # Let's assume Channel shape is (n_outputs, n_inputs) based on "prob of i given j"
    
    # Joint distribution p(x, y) = p(y|x) * p(x)
    # P(Y=i, X=j) = Channel[i][j] * Prior[j]
    
    # We need to broadcast prior to match channel shape
    # Channel: rows i, cols j
    # Prior: index j
    
    # joint[i, j] = channel[i, j] * prior[j]
    joint = channel * prior[np.newaxis, :]
    
    # Marginal p(y) = sum_x p(x, y) = sum_j joint[i, j] (sum over inputs)
    py = np.sum(joint, axis=1)
    
    # Marginal p(x) = prior
    px = prior
    
    # Mutual Info I(X;Y) = sum_x sum_y p(x,y) log2( p(x,y) / (p(x)p(y)) )
    # Handle zeros: if p(x,y) is 0, term is 0.
    
    mi = 0.0
    rows, cols = joint.shape
    for i in range(rows): # y
        for j in range(cols): # x
            pxy = joint[i, j]
            if pxy > 1e-12:
                # p(x) = prior[j]
                # p(y) = py[i]
                mi += pxy * np.log2(pxy / (prior[j] * py[i]))
                
    return mi

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

    print("All tests passed for Subproblem 2!")

if __name__ == "__main__":
    test_mutual_info()

