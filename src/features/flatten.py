"""Flatten extractor: converts 3D EEG epochs to 2D feature arrays."""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin


class FlattenExtractor(BaseEstimator, TransformerMixin):
    """
    Flatten EEG epochs to 2D array.

    Converts (n_epochs, n_channels, n_times) to (n_epochs, n_channels * n_times)
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: NDArray[np.float64],
            y: Optional[NDArray[np.int64]] = None) -> 'FlattenExtractor':
        """Fit method (nothing to fit)."""
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Flatten EEG epochs.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times)

        Returns
        -------
        X_flat : np.ndarray
            Flattened data of shape (n_epochs, n_channels * n_times)
        """
        n_epochs = X.shape[0]
        return X.reshape(n_epochs, -1)
