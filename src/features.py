"""
Feature extraction module for EEG data.

Implements various feature extraction methods:
- Power Spectral Density (PSD) using Welch method
- Band power extraction
- Log variance features
"""

from typing import Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

from constants import (
    EPSILON,
    EEG_SAMPLING_RATE,
    WELCH_NPERSEG,
    WELCH_NOVERLAP,
    MU_BAND_LOW,
    MU_BAND_HIGH,
    BETA_BAND_LOW,
    BETA_BAND_HIGH,
    TEST_N_EPOCHS,
    TEST_N_CHANNELS,
    TEST_N_TIMES,
    FILTER_EDGE_LOW,
    FILTER_EDGE_HIGH,
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

        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                # Compute PSD using Welch's method
                freqs, psd = signal.welch(X[epoch_idx, ch_idx, :],
                                          fs=self.fs,
                                          nperseg=min(self.nperseg, n_times),
                                          noverlap=min(self.noverlap, n_times // 2))

                # Extract band powers
                for band_idx, (band_name, (low, high)) in enumerate(self.freq_bands.items()):
                    # Find frequency indices within the band
                    idx_band = np.logical_and(freqs >= low, freqs <= high)
                    # Mean power in the band
                    band_power = np.mean(psd[idx_band])
                    features[epoch_idx, ch_idx * n_bands + band_idx] = band_power

        return features


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
            # Design bandpass filter
            nyq = self.fs / 2
            low_norm = low / nyq
            high_norm = high / nyq

            # Handle edge cases
            if low_norm <= 0:
                low_norm = FILTER_EDGE_LOW
            if high_norm >= 1:
                high_norm = FILTER_EDGE_HIGH

            b, a = signal.butter(4, [low_norm, high_norm], btype='band')

            for epoch_idx in range(n_epochs):
                for ch_idx in range(n_channels):
                    # Apply bandpass filter
                    filtered = signal.filtfilt(b, a, X[epoch_idx, ch_idx, :])
                    # Compute variance (power)
                    power = np.var(filtered)
                    features[epoch_idx, ch_idx * n_bands + band_idx] = power

        return features


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
            # (n_epochs, n_channels, n_times) -> variance per channel
            variances = np.var(X, axis=2)
        else:
            # Already (n_epochs, n_features)
            variances = X

        # Log transform (add small epsilon to avoid log(0))
        log_var = np.log(variances + EPSILON)

        return log_var


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


if __name__ == "__main__":
    # Test feature extractors
    print("Testing feature extractors...")

    # Create dummy EEG data
    n_epochs, n_channels, n_times = TEST_N_EPOCHS, TEST_N_CHANNELS, TEST_N_TIMES
    X = np.random.randn(n_epochs, n_channels, n_times)

    # Test PSD extractor
    psd = PSDExtractor(fs=EEG_SAMPLING_RATE)
    X_psd = psd.fit_transform(X)
    print(f"PSD features shape: {X_psd.shape}")

    # Test band power extractor
    bp = BandPowerExtractor(fs=EEG_SAMPLING_RATE)
    X_bp = bp.fit_transform(X)
    print(f"Band power features shape: {X_bp.shape}")

    # Test log variance
    lv = LogVarianceExtractor()
    X_lv = lv.fit_transform(X)
    print(f"Log variance features shape: {X_lv.shape}")

    # Test flatten
    flat = FlattenExtractor()
    X_flat = flat.fit_transform(X)
    print(f"Flattened features shape: {X_flat.shape}")
