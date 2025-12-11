"""
Custom PCA (Principal Component Analysis) implementation.

An alternative dimensionality reduction method for EEG data.
"""

from typing import Optional
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin


class MyPCA(BaseEstimator, TransformerMixin):
    """
    Custom PCA implementation as sklearn transformer.

    An alternative dimensionality reduction method.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep

    Attributes
    ----------
    mean_ : np.ndarray
        Per-feature mean of the training data
    components_ : np.ndarray
        Principal axes in feature space (n_features, n_components)
    explained_variance_ : np.ndarray
        Variance explained by each component
    explained_variance_ratio_ : np.ndarray
        Percentage of variance explained by each component
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
