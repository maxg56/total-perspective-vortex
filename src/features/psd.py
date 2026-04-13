"""Power Spectral Density feature extractor using Welch's method."""

from typing import Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

from constants import (
    EEG_SAMPLING_RATE,
    WELCH_NPERSEG,
    WELCH_NOVERLAP,
    MU_BAND_LOW,
    MU_BAND_HIGH,
    BETA_BAND_LOW,
    BETA_BAND_HIGH,
)


class PSDExtractor(BaseEstimator, TransformerMixin):
    """
    Extract Power Spectral Density features using Welch's method.

    Parameters
    ----------
    fs : float
        Sampling frequency of the EEG signal
    nperseg : int
        Length of each segment for Welch's method
    noverlap : int
        Number of overlapping points between segments
    freq_bands : dict
        Dictionary mapping band names to (low, high) frequency tuples
    """

    def __init__(self, fs: float = EEG_SAMPLING_RATE, nperseg: int = WELCH_NPERSEG,
                 noverlap: int = WELCH_NOVERLAP,
                 freq_bands: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.freq_bands = freq_bands or {
            'mu': (MU_BAND_LOW, MU_BAND_HIGH),
            'beta': (BETA_BAND_LOW, BETA_BAND_HIGH),
        }

    def fit(self, X: NDArray[np.float64],
            y: Optional[NDArray[np.int64]] = None) -> 'PSDExtractor':
        """Fit method (nothing to fit for PSD extraction)."""
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Extract PSD features from EEG epochs.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times)

        Returns
        -------
        features : np.ndarray
            PSD features of shape (n_epochs, n_channels * n_bands)
        """
        n_epochs, n_channels, n_times = X.shape
        n_bands = len(self.freq_bands)
        features = np.zeros((n_epochs, n_channels * n_bands))

        # Compute Welch PSD for all epochs and channels at once (axis=-1)
        # freqs: (n_freqs,), psd: (n_epochs, n_channels, n_freqs)
        freqs, psd = signal.welch(
            X, fs=self.fs,
            nperseg=min(self.nperseg, n_times),
            noverlap=min(self.noverlap, n_times // 2),
            axis=-1
        )

        for band_idx, (band_name, (low, high)) in enumerate(self.freq_bands.items()):
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Mean over freq bins -> (n_epochs, n_channels)
            band_power = psd[:, :, idx_band].mean(axis=-1)
            # Layout: features[:, ch * n_bands + band_idx] for all ch
            features[:, band_idx::n_bands] = band_power

        return features
