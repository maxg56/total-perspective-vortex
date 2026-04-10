"""
Custom PCA (Principal Component Analysis) implementation.

An alternative dimensionality reduction method for EEG data.
"""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from constants import DEFAULT_N_COMPONENTS_PCA, EPSILON
from transforms.linalg import my_eigh


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

    def __init__(self, n_components: int = DEFAULT_N_COMPONENTS_PCA) -> None:
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
        n_samples, n_features = X.shape

        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        if n_features <= n_samples:
            # Standard: eigendecompose the covariance matrix
            cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)
            eigenvalues, eigenvectors = my_eigh(cov)

            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            eigenvectors = eigenvectors[:, sorted_idx]

            self.components_ = eigenvectors[:, :self.n_components]
        else:
            # Dual trick: eigendecompose X X^T (n_samples x n_samples)
            # when n_features >> n_samples, this is much faster
            gram = np.dot(X_centered, X_centered.T) / (n_samples - 1)
            eigenvalues, U = my_eigh(gram)

            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            U = U[:, sorted_idx]

            # Recover eigenvectors in feature space: v = X^T u / (sqrt(lambda * (n-1)))
            components = np.zeros((n_features, min(n_samples, self.n_components)))
            for i in range(min(n_samples, self.n_components)):
                if eigenvalues[i] > EPSILON:
                    v = X_centered.T @ U[:, i]
                    v = v / (np.sqrt(eigenvalues[i] * (n_samples - 1))
                             + EPSILON)
                    components[:, i] = v
            self.components_ = components[:, :self.n_components]

        self.explained_variance_ = eigenvalues[:self.n_components]
        total_var = np.sum(np.maximum(eigenvalues, 0.0))
        self.explained_variance_ratio_ = (
            self.explained_variance_ / (total_var + EPSILON)
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
