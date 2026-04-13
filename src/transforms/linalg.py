"""
Custom linear algebra implementations.

Provides hand-coded replacements for scipy/numpy linear algebra functions:
- Covariance matrix computation
- Cholesky decomposition
- Jacobi eigenvalue algorithm for symmetric matrices
- Generalized eigenvalue problem solver
- SVD via eigendecomposition

These implementations avoid relying on scipy.linalg for core decompositions,
demonstrating the underlying mathematical algorithms.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from constants import EPSILON, EPSILON_SMALL


def my_covariance(X: NDArray[np.float64],
                  rowvar: bool = True) -> NDArray[np.float64]:
    """
    Compute the covariance matrix from scratch.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. If rowvar=True, each row is a variable and columns are
        observations. If rowvar=False, each column is a variable.
    rowvar : bool
        If True, rows are variables (default). If False, columns are variables.

    Returns
    -------
    cov : np.ndarray
        Covariance matrix of shape (n_vars, n_vars)
    """
    if not rowvar:
        X = X.T

    n_vars, n_obs = X.shape

    # Center the data (subtract mean of each variable)
    mean = np.zeros(n_vars)
    for i in range(n_vars):
        s = 0.0
        for j in range(n_obs):
            s += X[i, j]
        mean[i] = s / n_obs

    X_centered = np.empty_like(X)
    for i in range(n_vars):
        for j in range(n_obs):
            X_centered[i, j] = X[i, j] - mean[i]

    # Compute covariance: cov = X_centered @ X_centered.T / (n_obs - 1)
    cov = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i, n_vars):
            s = 0.0
            for k in range(n_obs):
                s += X_centered[i, k] * X_centered[j, k]
            cov[i, j] = s / (n_obs - 1)
            cov[j, i] = cov[i, j]  # Symmetric

    return cov


def my_cholesky(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the Cholesky decomposition A = L @ L.T.

    Uses the Cholesky-Banachiewicz algorithm with vectorized
    dot products for the inner sums.

    Parameters
    ----------
    A : np.ndarray
        Symmetric positive-definite matrix of shape (n, n)

    Returns
    -------
    L : np.ndarray
        Lower triangular matrix of shape (n, n)

    Raises
    ------
    ValueError
        If the matrix is not positive definite
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        # Vectorized dot product for off-diagonal elements
        for j in range(i):
            s = np.dot(L[i, :j], L[j, :j])
            L[i, j] = (A[i, j] - s) / L[j, j]

        # Diagonal element
        s = np.dot(L[i, :i], L[i, :i])
        val = A[i, i] - s
        if val <= 0:
            raise ValueError(
                "Matrix is not positive definite "
                f"(diagonal element {i} = {val})")
        L[i, i] = np.sqrt(val)

    return L


def _solve_lower_triangular(
    L: NDArray[np.float64],
    b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Solve L @ x = b where L is lower triangular (forward substitution).

    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix of shape (n, n)
    b : np.ndarray
        Right-hand side vector of shape (n,)

    Returns
    -------
    x : np.ndarray
        Solution vector of shape (n,)
    """
    n = L.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s = np.dot(L[i, :i], x[:i])
        x[i] = (b[i] - s) / L[i, i]

    return x


def _solve_upper_triangular(
    U: NDArray[np.float64],
    b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Solve U @ x = b where U is upper triangular (back substitution).

    Parameters
    ----------
    U : np.ndarray
        Upper triangular matrix of shape (n, n)
    b : np.ndarray
        Right-hand side vector of shape (n,)

    Returns
    -------
    x : np.ndarray
        Solution vector of shape (n,)
    """
    n = U.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        s = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (b[i] - s) / U[i, i]

    return x


def _invert_lower_triangular(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the inverse of a lower triangular matrix.

    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix of shape (n, n)

    Returns
    -------
    L_inv : np.ndarray
        Inverse of L, also lower triangular
    """
    n = L.shape[0]
    L_inv = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        L_inv[:, j] = _solve_lower_triangular(L, e_j)

    return L_inv


def my_eigh(A: NDArray[np.float64],
            max_iter: int = 200) -> Tuple[NDArray[np.float64],
                                          NDArray[np.float64]]:
    """
    Compute eigenvalues and eigenvectors of a symmetric matrix using the Jacobi algorithm.

    The Jacobi method iteratively applies Givens rotations to
    diagonalize the matrix. Each rotation zeros out an off-diagonal
    element. Uses vectorized numpy operations for performance.

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix of shape (n, n)
    max_iter : int
        Maximum number of sweep iterations (default: 200)

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues in ascending order (n,)
    eigenvectors : np.ndarray
        Corresponding eigenvectors as columns (n, n)
    """
    n = A.shape[0]
    S = A.astype(np.float64).copy()
    V = np.eye(n, dtype=np.float64)

    tol = EPSILON * n * n

    for _ in range(max_iter):
        # Compute sum of squares of off-diagonal elements (vectorized)
        off_diag = np.triu(S, k=1)
        off_diag_sum: float = float(np.sum(off_diag ** 2))

        if off_diag_sum < tol:
            break

        # One sweep: iterate over all upper-triangle pairs
        for p in range(n):
            for q in range(p + 1, n):
                if abs(S[p, q]) < EPSILON:
                    continue

                # Compute Jacobi rotation parameters
                if abs(S[p, p] - S[q, q]) < EPSILON:
                    theta = np.pi / 4.0
                else:
                    tau = (S[q, q] - S[p, p]) / (2.0 * S[p, q])
                    if tau >= 0:
                        t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
                    else:
                        t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
                    theta = np.arctan(t)

                c = np.cos(theta)
                s = np.sin(theta)

                # Apply Givens rotation using vectorized column operations
                # Update columns p and q of S
                S_p = S[:, p].copy()
                S_q = S[:, q].copy()
                S[:, p] = c * S_p - s * S_q
                S[:, q] = s * S_p + c * S_q

                # Update rows p and q of S (symmetric)
                S_p = S[p, :].copy()
                S_q = S[q, :].copy()
                S[p, :] = c * S_p - s * S_q
                S[q, :] = s * S_p + c * S_q

                # Correct the (p,q) and (q,p) elements to be exactly 0
                S[p, q] = 0.0
                S[q, p] = 0.0

                # Accumulate eigenvectors (vectorized column update)
                V_p = V[:, p].copy()
                V_q = V[:, q].copy()
                V[:, p] = c * V_p - s * V_q
                V[:, q] = s * V_p + c * V_q

    eigenvalues = np.diag(S)

    # Sort in ascending order (like scipy.linalg.eigh)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    return eigenvalues, V


def my_eigh_generalized(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    max_iter: int = 200
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Solve the generalized eigenvalue problem A @ x = lambda * B @ x.

    Reduces to standard form using Cholesky decomposition of B:
        B = L @ L^T
        L^{-1} A L^{-T} y = lambda y
        x = L^{-T} y

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix of shape (n, n)
    B : np.ndarray
        Symmetric positive-definite matrix of shape (n, n)
    max_iter : int
        Maximum Jacobi iterations

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues in ascending order (n,)
    eigenvectors : np.ndarray
        Generalized eigenvectors as columns (n, n)
    """
    n = A.shape[0]

    # Regularize B to ensure positive definiteness
    B_reg = B + EPSILON_SMALL * np.eye(n)

    # Cholesky decomposition: B = L @ L^T
    L = my_cholesky(B_reg)
    L_inv = _invert_lower_triangular(L)

    # Transform to standard eigenvalue problem
    # C = L^{-1} A L^{-T}
    C = L_inv @ A @ L_inv.T

    # Ensure symmetry (numerical precision)
    C = (C + C.T) / 2.0

    # Solve standard eigenvalue problem
    eigenvalues, Y = my_eigh(C, max_iter=max_iter)

    # Back-transform eigenvectors: x = L^{-T} y
    eigenvectors = L_inv.T @ Y

    return eigenvalues, eigenvectors


def my_svd(A: NDArray[np.float64],
           max_iter: int = 200) -> Tuple[NDArray[np.float64],
                                         NDArray[np.float64],
                                         NDArray[np.float64]]:
    """
    Compute the Singular Value Decomposition A = U @ diag(s) @ Vt.

    Uses eigendecomposition of A^T @ A to find V and singular values,
    then computes U = A @ V @ diag(1/s). When m < n, uses the dual
    trick (eigendecompose A @ A^T instead) for efficiency.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n)
    max_iter : int
        Maximum Jacobi iterations

    Returns
    -------
    U : np.ndarray
        Left singular vectors of shape (m, k) where k = min(m, n)
    s : np.ndarray
        Singular values of shape (k,), in descending order
    Vt : np.ndarray
        Right singular vectors (transposed) of shape (k, n)
    """
    m, n = A.shape
    k = min(m, n)

    if m >= n:
        # Standard approach: eigendecompose A^T A
        AtA = A.T @ A
        AtA = (AtA + AtA.T) / 2.0
        eigenvalues, V = my_eigh(AtA, max_iter=max_iter)

        eigenvalues = eigenvalues[::-1]
        V = V[:, ::-1]
        eigenvalues = eigenvalues[:k]
        V = V[:, :k]

        s = np.sqrt(np.maximum(eigenvalues, 0.0))

        U = np.zeros((m, k), dtype=np.float64)
        for i in range(k):
            if s[i] > EPSILON:
                U[:, i] = (A @ V[:, i]) / s[i]
        Vt = V.T
    else:
        # Dual trick: eigendecompose A A^T (smaller matrix)
        AAt = A @ A.T
        AAt = (AAt + AAt.T) / 2.0
        eigenvalues, U = my_eigh(AAt, max_iter=max_iter)

        eigenvalues = eigenvalues[::-1]
        U = U[:, ::-1]
        eigenvalues = eigenvalues[:k]
        U = U[:, :k]

        s = np.sqrt(np.maximum(eigenvalues, 0.0))

        V = np.zeros((n, k), dtype=np.float64)
        for i in range(k):
            if s[i] > EPSILON:
                V[:, i] = (A.T @ U[:, i]) / s[i]
        Vt = V.T

    return U, s, Vt
