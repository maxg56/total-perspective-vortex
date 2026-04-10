"""Wavelet feature extractor using a custom Morlet CWT implementation."""

from typing import Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from constants import (
    EEG_SAMPLING_RATE,
    MU_BAND_LOW,
    MU_BAND_HIGH,
    BETA_BAND_LOW,
    BETA_BAND_HIGH,
)


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
    n = np.arange(n_points) - n_points // 2
    eta = n / scale
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

    x_fft = np.fft.fft(x)
    coefficients: NDArray[np.float64] = np.zeros((n_scales, n_times), dtype=np.float64)

    for i, scale in enumerate(scales):
        wavelet = _morlet_wavelet(n_times, scale, fs, omega0)
        wavelet_fft = np.fft.fft(wavelet)
        conv = np.fft.ifft(x_fft * np.conj(wavelet_fft))
        coefficients[i, :] = np.abs(conv)

    return coefficients


def _freq_to_scale(freq: float, fs: float, omega0: float = 5.0) -> float:
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

        band_scales = {}
        for band_name, (low, high) in self.freq_bands.items():
            freqs = np.linspace(low, high, self.n_scales_per_band)
            scales = np.array([_freq_to_scale(f, self.fs, self.omega0) for f in freqs])
            band_scales[band_name] = scales

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
                cwt_coeffs = _cwt_morlet(sig, all_scales, self.fs, self.omega0)

                for band_idx, (s_start, s_end) in enumerate(band_boundaries):
                    band_energy = np.mean(cwt_coeffs[s_start:s_end, :] ** 2)
                    features[epoch_idx, ch_idx * n_bands + band_idx] = band_energy

        return features
