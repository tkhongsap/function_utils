# Problem 03: Gauss-Seidel Method

## Overview

| Field | Value |
|-------|-------|
| **Domain** | Numerical Linear Algebra |
| **Task** | Solve $Ax = b$ using Gauss-Seidel iteration |
| **Dependency** | `numpy` only |
| **Status** | In Review |

---

## Problem Statement

Implement an iterative solver for linear systems using matrix splitting $A = M - N$ where $M = D - L$.

- $D$ = diagonal of $A$
- $L$ = lower triangular of $A$

---

## Iteration Formula

$$x_{i}^{(k+1)} = \frac{b_i - \sum_{j>i} a_{ij}x_j^{(k)} - \sum_{j<i} a_{ij} x_j^{(k+1)}}{a_{ii}}$$

**Stopping criterion:** $\|x_k - x_{k-1}\|_2 < \epsilon$

---

## I/O Specification

### Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | N×N array | Coefficient matrix |
| `b` | N×1 array | Right-hand side vector |
| `eps` | float | Error tolerance |
| `x_true` | N×1 array | True solution (for error calc) |
| `x0` | N×1 array | Initial guess (zeros) |

### Output

| Parameter | Description |
|-----------|-------------|
| `residual` | $\|Ax - b\|_2$ |
| `errors` | $\|x - x_{true}\|_2$ |

---

## Convergence Guarantee

Converges when $A$ is:

- Diagonally dominant, **OR**
- Symmetric positive definite

---

## Key Evaluation Points

Before reviewing subquestions, check:

- [ ] Is the iteration formula complete?
- [ ] Are stopping criteria clear?
- [ ] Can this be implemented with **numpy only**?
- [ ] Are all I/O types specified?
