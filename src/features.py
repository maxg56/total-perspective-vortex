"""
Feature extraction module for EEG data.

Implements various feature extraction methods:
- Power Spectral Density (PSD) using Welch method
- Band power extraction
- Log variance features
- Wavelet transform (custom Morlet CWT implementation)
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


def _morlet_wavelet(n_points: int, scale: float, fs: float,
                    omega0: float = 5.0) -> NDArray[np.float64]:
    """
    Generate a Morlet wavelet at a given scale.

    The Morlet wavelet is defined as:
        psi(t) = pi^{-1/4} * exp(i * omega0 * t) * exp(-t^2 / 2)

    Parameters
    ----------
    n_points : int
        Number of time points for the wavelet
    scale : float
        Scale parameter (controls frequency resolution)
    fs : float
        Sampling frequency
    omega0 : float
        Central frequency of the wavelet (default: 5.0)

    Returns
    -------
    wavelet : np.ndarray
        Complex Morlet wavelet of shape (n_points,)
    """
    # Sample indices centered at 0
    n = np.arange(n_points) - n_points // 2
    # Normalized time: scale is in sample units (from _freq_to_scale)
    eta = n / scale
    # Morlet wavelet with normalization
    norm = (np.pi ** (-0.25)) / np.sqrt(scale)
    wavelet = norm * np.exp(1j * omega0 * eta) * np.exp(-eta ** 2 / 2.0)
    return wavelet


def _cwt_morlet(x: NDArray[np.float64], scales: NDArray[np.float64],
                fs: float, omega0: float = 5.0) -> NDArray[np.float64]:
    """
    Compute the Continuous Wavelet Transform using Morlet wavelets.

    Implements CWT via convolution in the frequency domain (FFT-based)
    for efficiency.

    Parameters
    ----------
    x : np.ndarray
        Input signal of shape (n_times,)
    scales : np.ndarray
        Array of wavelet scales
    fs : float
        Sampling frequency
    omega0 : float
        Morlet wavelet central frequency

    Returns
    -------
    coefficients : np.ndarray
        CWT coefficients of shape (n_scales, n_times)
    """
    n_times = len(x)
    n_scales = len(scales)

    # FFT of the input signal
    x_fft = np.fft.fft(x)

    coefficients: NDArray[np.float64] = np.zeros((n_scales, n_times), dtype=np.float64)

    for i, scale in enumerate(scales):
        # Generate wavelet in time domain, then FFT
        wavelet = _morlet_wavelet(n_times, scale, fs, omega0)
        wavelet_fft = np.fft.fft(wavelet)

        # Convolution via FFT multiplication
        conv = np.fft.ifft(x_fft * np.conj(wavelet_fft))
        coefficients[i, :] = np.abs(conv)

    return coefficients


def _freq_to_scale(freq: float, fs: float,
                   omega0: float = 5.0) -> float:
    """
    Convert a frequency to the corresponding Morlet wavelet scale.

    For the Morlet wavelet, the relationship between scale and
    pseudo-frequency is: f = omega0 / (2 * pi * scale)

    Parameters
    ----------
    freq : float
        Target frequency in Hz
    fs : float
        Sampling frequency
    omega0 : float
        Morlet wavelet central frequency

    Returns
    -------
    scale : float
        Corresponding wavelet scale
    """
    return float(omega0 / (2.0 * np.pi * freq) * fs)


class WaveletExtractor(BaseEstimator, TransformerMixin):
    """
    Extract wavelet transform features from EEG data.

    Uses a custom Continuous Wavelet Transform (CWT) implementation
    with Morlet wavelets to compute time-frequency energy features
    in physiologically relevant EEG frequency bands (mu and beta).

    Parameters
    ----------
    fs : float
        Sampling frequency of the EEG signal
    freq_bands : dict or None
        Dictionary mapping band names to (low, high) frequency tuples.
        Default: mu (8-12 Hz) and beta (12-30 Hz) bands.
    n_scales_per_band : int
        Number of wavelet scales to sample within each frequency band
    omega0 : float
        Morlet wavelet central frequency parameter
    """

    def __init__(self, fs: float = EEG_SAMPLING_RATE,
                 freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
                 n_scales_per_band: int = 5,
                 omega0: float = 5.0) -> None:
        self.fs = fs
        self.freq_bands = freq_bands or {
            'mu': (MU_BAND_LOW, MU_BAND_HIGH),
            'beta': (BETA_BAND_LOW, BETA_BAND_HIGH),
        }
        self.n_scales_per_band = n_scales_per_band
        self.omega0 = omega0

    def fit(self, X: NDArray[np.float64],
            y: Optional[NDArray[np.int64]] = None) -> 'WaveletExtractor':
        """Fit method (nothing to fit for wavelet extraction)."""
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Extract wavelet energy features from EEG epochs.

        For each epoch and channel, computes the CWT using Morlet wavelets
        at scales corresponding to the target frequency bands. Returns the
        mean energy (squared magnitude) of the CWT coefficients within each
        band.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times)

        Returns
        -------
        features : np.ndarray
            Wavelet features of shape (n_epochs, n_channels * n_bands)
        """
        n_epochs, n_channels, n_times = X.shape
        n_bands = len(self.freq_bands)
        features = np.zeros((n_epochs, n_channels * n_bands))

        # Pre-compute scales for each band
        band_scales = {}
        for band_name, (low, high) in self.freq_bands.items():
            freqs = np.linspace(low, high, self.n_scales_per_band)
            scales = np.array([
                _freq_to_scale(f, self.fs, self.omega0)
                for f in freqs
            ])
            band_scales[band_name] = scales

        # Concatenate all scales for a single CWT pass
        all_scales = np.concatenate(list(band_scales.values()))
        band_boundaries = []
        start = 0
        for band_name in self.freq_bands:
            n_sc = len(band_scales[band_name])
            band_boundaries.append((start, start + n_sc))
            start += n_sc

        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                sig = X[epoch_idx, ch_idx, :]

                # Compute CWT for all scales at once
                cwt_coeffs = _cwt_morlet(
                    sig, all_scales, self.fs, self.omega0)

                # Extract mean energy per band
                for band_idx, (s_start, s_end) in enumerate(
                        band_boundaries):
                    band_energy = np.mean(
                        cwt_coeffs[s_start:s_end, :] ** 2)
                    feat_idx = ch_idx * n_bands + band_idx
                    features[epoch_idx, feat_idx] = band_energy

        return features


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
