"""Log-variance feature extractor, commonly used after CSP spatial filtering."""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from constants import EPSILON


class LogVarianceExtractor(BaseEstimator, TransformerMixin):
    """
    Extract log-variance features from EEG data.

    This is commonly used after CSP spatial filtering.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: NDArray[np.float64],
            y: Optional[NDArray[np.int64]] = None) -> 'LogVarianceExtractor':
        """Fit method (nothing to fit)."""
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute log-variance features.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times)
            or (n_epochs, n_components) if already CSP transformed

        Returns
        -------
        features : np.ndarray
            Log-variance features
        """
        if X.ndim == 3:
            variances = np.var(X, axis=2)
        else:
            variances = X

        return np.log(variances + EPSILON)
