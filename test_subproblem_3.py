import numpy as np

def blahut_arimoto(channel, e):
    # Reference Implementation for generating targets
    # Inputs:
    # channel: (n_outputs, n_inputs)
    # e: error threshold
    
    channel = np.asarray(channel)
    n_outputs, n_inputs = channel.shape
    
    # Initialize prior uniformly
    p = np.ones(n_inputs) / n_inputs
    
    # Loop
    # Max iterations to prevent infinite loop if e is too small or logic fails
    max_iter = 10000
    
    rate_old = 0.0
    
    for _ in range(max_iter):
        # 1. Calculate c_x = exp( D(p(y|x) || p(y)) )
        # But standard BA update is simpler:
        # q(y) = sum_x p(x) p(y|x)
        # phi(x|y) = p(x) p(y|x) / q(y)
        # p_new(x) = exp( sum_y p(y|x) log phi(x|y) ) / normalization ?? No.
        
        # Standard Algorithm (Maximizing I):
        # Step 1: Compute q(y) output distribution based on current p(x)
        # q(y) = sum_j p(j) C(i|j)  (where C[i][j] = p(i|j))
        # q = channel @ p
        
        # Step 2: Maximize I w.r.t p(x) for fixed q(y)? No, that's alternating min.
        # Update rule:
        # p_new(j) = p(j) * exp( D( p(y|j) || q(y) ) ) / normalization
        # D( p(y|j) || q(y) ) = sum_i C[i,j] * log( C[i,j] / q[i] )
        
        # Let's use Natural Logarithm for the update step (standard BA)
        # But we want result in Bits?
        # If we use log2 in D, we assume exp means 2^x? No, exp(x) is e^x.
        # If formula is p_new = p_old * exp(D), and we want it to work, D must be in natural units (nats).
        # So we use np.log (base e).
        
        q = channel @ p
        
        # Avoid division by zero in q (though q should be >0 if p>0 and channel connected)
        # C[i,j] / q[i]. If q[i] is 0, then C[i,j] must be 0 for all j where p(j)>0. 0/0 -> 0 contribution.
        
        # Calculate D_j = KL( C[:,j] || q[:] ) for each input j
        # D_j = sum_i C[i,j] * log( C[i,j] / q[i] )
        
        D = np.zeros(n_inputs)
        for j in range(n_inputs):
            # Column j of channel
            col = channel[:, j]
            # KL divergence
            # Mask zeros
            mask = (col > 0) & (q > 0)
            if np.any(mask):
                D[j] = np.sum(col[mask] * np.log(col[mask] / q[mask]))
                
        # Update p
        # p_new(j) = p(j) * exp(D[j]) / sum(...)
        numerator = p * np.exp(D)
        p_new = numerator / np.sum(numerator)
        
        # Calculate Capacity (Mutual Information) in Bits
        # I(X;Y) = sum_j p(j) D_j(bits)
        # But D calculated above is in nats.
        # I_nats = sum(p * D)
        # I_bits = I_nats / log(2)
        
        rate_new = np.sum(p_new * D) / np.log(2) # Calculate rate using NEW p or OLD p?
        # Usually checking convergence of the capacity value.
        # Let's use p_new for the rate calculation.
        
        if np.abs(rate_new - rate_old) < e:
            return rate_new
            
        rate_old = rate_new
        p = p_new
        
    return rate_new

def test_blahut_arimoto():
    # Test Case 1: Channel 3x3
    np.random.seed(0)
    channel1 = np.array([[1,0,1/4],[0,1,1/4],[0,0,1/2]])
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
    target4 = 1 - (-0.2*np.log2(0.2) - 0.8*np.log2(0.8))
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
    
    print("All tests passed for Subproblem 3!")

if __name__ == "__main__":
    test_blahut_arimoto()

