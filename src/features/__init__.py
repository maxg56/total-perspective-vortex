"""
Feature extraction package for EEG data.

Implements various feature extraction methods:
- Power Spectral Density (PSD) using Welch method
- Band power extraction
- Log variance features
- Wavelet transform (custom Morlet CWT implementation)
"""

from features.psd import PSDExtractor
from features.bandpower import BandPowerExtractor
from features.logvariance import LogVarianceExtractor
from features.flatten import FlattenExtractor
from features.wavelet import WaveletExtractor

__all__ = [
    'PSDExtractor',
    'BandPowerExtractor',
    'LogVarianceExtractor',
    'FlattenExtractor',
    'WaveletExtractor',
]
