

12/12/2025, 09:34                                                                Reviewer Dashboard: Task Review - Airtable

# 29_29.3

**Subquestion**  
With the previous functions, provide a function that performs Gram-Schmidt orthogonalization on N linearly independent vectors in N-dimension space. The input is an $N \times N$ numpy array, containing N vectors in the shape of $N \times 1$. The output should also be an $N \times N$ numpy array, containing the orthogonal and normalized vectors.

**Function Header**  
```python
def orthogonalize(A):
    '''Perform Gram-Schmidt orthogonalization on the input vectors to produce orthogonal and normalized vectors.
    Input:
    A (N*N numpy array): N linearly independent vectors in the N-dimension space.
    Output:
    B (N*N numpy array): The collection of the orthonomal vectors.
    '''
```

**Return Line**  
```python
return B
```

**Test Cases**  
```python
[
    "A = np.array([[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, -1, 0]]).T\nB = orthogonalize(A)\nassert (np.isclose(0,B[:,0].T @ B[:,3])) == target",
    "A = np.array([[0, 1, 1], [1, 0, -1], [1, -1, 0]]).T\nassert np.allclose(orthogonalize(A), target)",
    "A = np.array([[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, -1, 0]]).T\nassert np.allclose(orthogonalize(A), target)",
    "A = np.array([[0, 1, 1, -1, 1], [1, 0, -1, 1, -1], [1, -1, 0, -1, -1], [-1, 1, -1, 0, -1], [-1, -1, -1, -1, 0]]).T\nassert np.allclose(orthogonalize(A), target)"
]
```

**Answer**  
```python
def orthogonalize(A):
    """
    Perform Gram-Schmidt orthogonalization on the input vectors to produce orthogonal and normalized vectors.
    Input:
    A (N*N numpy array): N linearly independent vectors in the N-dimension space.
    Output:
    B (N*N numpy array): The collection of the orthonomal vectors.
    """
    B = np.zeros(A.shape)
    N = A.shape[-1]
    for j in range(N):
        B[:,j] = A[:,j]
        for jj in range(j):
            B[:,j] -= B[:,jj]*inner_product(A[:,j],B[:,jj])
        B[:,j] = normalize(B[:,j])
    return B
```

----

## Expert 1's Answer

**Overall Classification**  
Ineffective

**Task Issues**  
Test unfairness

**Rationale**  
A variable called "target" was provided for the test cases, but was not defined beforehand. Therefore, we do not know that this value must be, which makes the problem's test cases unfair.



---


12/12/2025, 09:34    Reviewer Dashboard: Task Review - Airtable

# Expert 2's Answer

**Overall Classification**  
Effective

**Rationale**  
The subproblem is effective because the function produces the projection coefficient and subtracts the projection onto the previous orthogonal vectors. And each orthogonal vector is normalised

----

# Reviewer's Comments

**Overall Classification**  
–

**Rationale**  
–



---


12/12/2025, 09:34    Reviewer Dashboard: Task Review - Airtable

NO_CONTENT_HERE

https://airtable.com/appZUY7r3DxCmHBsG/pagljVAvVZdHiFBIi?detail=eyJwYWdlSWQiOiJwYWdRT21lRUpVd0lES2YxUyIsInJvd0lkIjoicmVjMzZ6aDFqZjVVenZPWFMiLCJzaG93Q29tbWVudHMiOmZhbHNlLCJxdW…  3/3
