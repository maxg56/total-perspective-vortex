"""Band power feature extractor using bandpass filtering and variance."""

from typing import Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

from constants import (
    EEG_SAMPLING_RATE,
    MU_BAND_LOW,
    MU_BAND_HIGH,
    BETA_BAND_LOW,
    BETA_BAND_HIGH,
    FILTER_EDGE_LOW,
    FILTER_EDGE_HIGH,
)


class BandPowerExtractor(BaseEstimator, TransformerMixin):
    """
    Extract band power features using bandpass filtering and variance.

    Parameters
    ----------
    fs : float
        Sampling frequency
    freq_bands : dict
        Dictionary mapping band names to (low, high) frequency tuples
    """

    def __init__(self, fs: float = EEG_SAMPLING_RATE,
                 freq_bands: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.fs = fs
        self.freq_bands = freq_bands or {
            'mu': (MU_BAND_LOW, MU_BAND_HIGH),
            'beta': (BETA_BAND_LOW, BETA_BAND_HIGH),
        }

    def fit(self, X: NDArray[np.float64],
            y: Optional[NDArray[np.int64]] = None) -> 'BandPowerExtractor':
        """Fit method (nothing to fit)."""
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Extract band power features.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times)

        Returns
        -------
        features : np.ndarray
            Band power features of shape (n_epochs, n_channels * n_bands)
        """
        n_epochs, n_channels, n_times = X.shape
        n_bands = len(self.freq_bands)
        features = np.zeros((n_epochs, n_channels * n_bands))

        for band_idx, (band_name, (low, high)) in enumerate(self.freq_bands.items()):
            nyq = self.fs / 2
            low_norm = low / nyq
            high_norm = high / nyq

            if low_norm <= 0:
                low_norm = FILTER_EDGE_LOW
            if high_norm >= 1:
                high_norm = FILTER_EDGE_HIGH

            b, a = signal.butter(4, [low_norm, high_norm], btype='band')

            for epoch_idx in range(n_epochs):
                for ch_idx in range(n_channels):
                    filtered = signal.filtfilt(b, a, X[epoch_idx, ch_idx, :])
                    power = np.var(filtered)
                    features[epoch_idx, ch_idx * n_bands + band_idx] = power

        return features
