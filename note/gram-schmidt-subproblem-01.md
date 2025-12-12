
12/12/2025, 09:33    Reviewer Dashboard: Task Review - Airtable

# 29_29.1

**Subquestion**  
Provide a fucntion that normalizes the input vector. The input should be a numpy array and the output should be a numpy array with the same shape.

**Function Header**  
```python
def normalize(v):
    '''Normalize the input vector.
    Input:
    v (N*1 numpy array): The input vector.
    Output:
    n (N*1 numpy array): The normalized vector.
    '''
```

**Return Line**  
`return n`

**Test Cases**  
```python
[
    "v = np.array([3,4])\nassert np.allclose(normalize(v), target)", 
    "v = np.array([1,2,3,4,5,6,7,8]).reshape(4,2)\nassert np.allclose(normalize(v), target)", 
    "v = np.array([i for i in range(12)]).reshape(3,2,2)\nassert np.allclose(normalize(v), target)"
]
```

**Answer**  
```python
def normalize(v):
    """
    Normalize the input vector.
    Input:
    v (N*1 numpy array): The input vector.
    Output:
    n (N*1 numpy array): The normalized vector.
    """
    norm = np.linalg.norm(v)
    n = v/norm if norm else v
    return n
```

----

## Expert 1's Answer

**Overall Classification**  
Ineffective

**Task Issues**  
Problem ambiguity

**Rationale**  
The definition for the inner product space (which also defines the norm) for this problem is not specified. Usually, when working with these, we use the Euclidian inner product, or the dot product, but if the inner product being used is different, it is possible to get different results from the norm of a vector. Therefore, we must be explicitly told to use the Euclidian Inner Product here.

----

## Expert 2's Answer

**Overall Classification**  
Effective

</page_header>


---



12/12/2025, 09:33             Reviewer Dashboard: Task Review - Airtable

# Rationale
The subproblem is effective because the function (normalize(v)) computes the norm of each vector v. After that, each vector (v) is divided by its norm. In addition, the norm of a vector is zero; nothing occurs, and there is no division.

----

## Reviewer's Comments

| Overall Classification | – |
|------------------------|---|
| Rationale              | – |



---


12/12/2025, 09:33    Reviewer Dashboard: Task Review - Airtable

NO_CONTENT_HERE

https://airtable.com/appZUY7r3DxCmHBsG/pagljVAvVZdHiFBIi?detail=eyJwYWdlSWQiOiJwYWdRT21lRUpVd0lES2YxUyIsInJvd0lkIjoicmVjaVdwaUtCbXNpRVZFcU8iLCJzaG93Q29tbWVudHMiOmZhbHNlLCJxdWV…  3/3
