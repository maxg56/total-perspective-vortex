"""
Custom Common Spatial Patterns (CSP) implementation.

CSP is a spatial filtering technique used for discriminating between
two classes in EEG-based Brain-Computer Interfaces.

The algorithm finds spatial filters that maximize the variance of one class
while minimizing the variance of the other class.

Mathematical Background:
------------------------
Given two classes of EEG data, CSP solves the generalized eigenvalue problem:

    Σ₁ W = (Σ₁ + Σ₂) W Λ

where:
    - Σ₁, Σ₂ are the average covariance matrices of classes 1 and 2
    - W is the matrix of spatial filters (eigenvectors)
    - Λ is the diagonal matrix of eigenvalues

The eigenvectors corresponding to the largest and smallest eigenvalues
are the most discriminative spatial filters.
"""

from typing import Optional
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

from constants import EPSILON, EPSILON_SMALL


class MyCSP(BaseEstimator, TransformerMixin):
    """
    Custom Common Spatial Patterns (CSP) implementation as sklearn transformer.

    Parameters
    ----------
    n_components : int
        Number of CSP components to keep (default: 4)
        The algorithm will keep n_components/2 from each end of the spectrum
    reg : float or None
        Regularization parameter for covariance estimation (default: None)
        If not None, adds reg * trace(cov) * I to the covariance matrices
    log : bool
        If True, return log-variance of CSP features (default: True)
    norm_trace : bool
        If True, normalize covariance matrices by their trace (default: True)

    Attributes
    ----------
    W_ : np.ndarray
        The CSP spatial filters (n_channels, n_components)
    eigenvalues_ : np.ndarray
        The eigenvalues associated with the spatial filters
    """

    def __init__(self, n_components: int = 4, reg: Optional[float] = None,
                 log: bool = True, norm_trace: bool = True) -> None:
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the average covariance matrix for a set of epochs.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times)

        Returns
        -------
        cov : np.ndarray
            Average covariance matrix of shape (n_channels, n_channels)
        """
        n_epochs, n_channels, n_times = X.shape
        # Vectorized covariance computation for all epochs
        covs = np.einsum('ijk,ilk->ijl', X, X) / (n_times - 1)

        # Normalize by trace if requested (vectorized)
        if self.norm_trace:
            traces = np.trace(covs, axis1=1, axis2=2)
            covs = covs / traces[:, np.newaxis, np.newaxis]
        # Average across epochs
        avg_cov = np.mean(covs, axis=0)

        # Apply regularization if specified
        if self.reg is not None:
            avg_cov += self.reg * np.trace(avg_cov) * np.eye(n_channels)

        return avg_cov

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit CSP spatial filters to the data.

        Parameters
        ----------
        X : np.ndarray
            Training EEG data of shape (n_epochs, n_channels, n_times)
        y : np.ndarray
            Labels of shape (n_epochs,) with exactly 2 classes

        Returns
        -------
        self : MyCSP
            The fitted transformer
        """
        # Get unique classes
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")

        # Separate data by class
        X_class1 = X[y == classes[0]]
        X_class2 = X[y == classes[1]]

        # Compute average covariance matrices for each class
        cov1 = self._compute_covariance(X_class1)
        cov2 = self._compute_covariance(X_class2)

        # Composite covariance
        cov_composite = cov1 + cov2

        # Solve generalized eigenvalue problem: cov1 @ W = cov_composite @ W @ D
        # This is equivalent to finding eigenvectors of cov_composite^(-1) @ cov1
        try:
            eigenvalues, eigenvectors = linalg.eigh(cov1, cov_composite)
        except linalg.LinAlgError:
            # If decomposition fails, add regularization
            eigenvalues, eigenvectors = linalg.eigh(
                cov1 + EPSILON_SMALL * np.eye(cov1.shape[0]),
                cov_composite + EPSILON_SMALL * np.eye(cov_composite.shape[0])
            )

        # Sort eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Select components from both ends of the spectrum
        # (most discriminative for class 1 and class 2)
        n_pairs = self.n_components // 2
        idx = np.concatenate([
            np.arange(n_pairs),  # First n_pairs (high eigenvalues - class 1)
            np.arange(-n_pairs, 0)  # Last n_pairs (low eigenvalues - class 2)
        ])

        self.W_ = eigenvectors[:, idx]
        self.eigenvalues_ = eigenvalues[idx]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CSP spatial filters to the data.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times)

        Returns
        -------
        X_csp : np.ndarray
            CSP features of shape (n_epochs, n_components)
            If log=True, returns log-variance of spatially filtered signals
        """
        if not hasattr(self, 'W_'):
            raise RuntimeError("CSP not fitted. Call fit() first.")

        # Apply spatial filters: X_filtered = W.T @ X
        # Shape: (n_epochs, n_components, n_times)
        # Vectorized spatial filtering using einsum
        X_filtered = np.einsum('ck,nkt->nct', self.W_.T, X)
        # Compute variance of each component
        # Shape: (n_epochs, n_components)
        variances = np.var(X_filtered, axis=2)

        # Normalize variances (protect against division by zero)
        variance_sum = variances.sum(axis=1, keepdims=True)
        variances /= (variance_sum + EPSILON)

        if self.log:
            # Log-transform (common for CSP features)
            return np.log(variances + EPSILON)
        else:
            return variances

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit CSP and transform data in one step.

        Parameters
        ----------
        X : np.ndarray
            Training EEG data of shape (n_epochs, n_channels, n_times)
        y : np.ndarray
            Labels of shape (n_epochs,)

        Returns
        -------
        X_csp : np.ndarray
            CSP features of shape (n_epochs, n_components)
        """
        return self.fit(X, y).transform(X)


class MyPCA(BaseEstimator, TransformerMixin):
    """
    Custom PCA implementation as sklearn transformer.

    An alternative dimensionality reduction method.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep
    """

    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MyPCA':
        """
        Fit PCA to the data.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self : MyPCA
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(cov)

        # Sort in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Keep top n_components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / eigenvalues.sum()
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply PCA transformation.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features)

        Returns
        -------
        X_pca : np.ndarray
            Transformed data of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)


if __name__ == "__main__":
    # Test CSP implementation
    print("Testing CSP implementation...")

    # Create dummy EEG data with 2 classes
    np.random.seed(42)
    n_epochs = 100
    n_channels = 64
    n_times = 480

    # Simulate class 1 with higher variance in first channels
    X1 = np.random.randn(n_epochs // 2, n_channels, n_times)
    X1[:, :5, :] *= 2  # Higher variance in first 5 channels

    # Simulate class 2 with higher variance in last channels
    X2 = np.random.randn(n_epochs // 2, n_channels, n_times)
    X2[:, -5:, :] *= 2  # Higher variance in last 5 channels

    X = np.concatenate([X1, X2], axis=0)
    y = np.array([0] * (n_epochs // 2) + [1] * (n_epochs // 2))

    # Shuffle
    perm = np.random.permutation(n_epochs)
    X = X[perm]
    y = y[perm]

    # Test CSP
    csp = MyCSP(n_components=4, log=True)
    X_csp = csp.fit_transform(X, y)

    print(f"Input shape: {X.shape}")
    print(f"CSP output shape: {X_csp.shape}")
    print(f"CSP eigenvalues: {csp.eigenvalues_}")
    print(f"CSP filter shape: {csp.W_.shape}")

    # Test PCA
    X_flat = X.reshape(X.shape[0], -1)
    pca = MyPCA(n_components=10)
    X_pca = pca.fit_transform(X_flat)

    print(f"\nPCA input shape: {X_flat.shape}")
    print(f"PCA output shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
