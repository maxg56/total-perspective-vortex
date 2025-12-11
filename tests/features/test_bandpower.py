"""
Unit tests for BandPowerExtractor class.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from constants import EEG_SAMPLING_RATE


class TestBandPowerExtractor:
    """Tests for BandPowerExtractor class."""

    def test_init_default_params(self):
        """Test BandPowerExtractor initialization with default parameters."""
        from features import BandPowerExtractor

        extractor = BandPowerExtractor()
        assert extractor.fs == EEG_SAMPLING_RATE
        assert 'mu' in extractor.freq_bands
        assert 'beta' in extractor.freq_bands

    def test_init_custom_params(self):
        """Test BandPowerExtractor with custom parameters."""
        from features import BandPowerExtractor

        custom_bands = {'theta': (4, 8), 'alpha': (8, 13)}
        extractor = BandPowerExtractor(fs=256.0, freq_bands=custom_bands)

        assert extractor.fs == 256.0
        assert extractor.freq_bands == custom_bands

    def test_fit_returns_self(self, synthetic_eeg_data):
        """Test that fit returns self."""
        from features import BandPowerExtractor

        X, y = synthetic_eeg_data
        extractor = BandPowerExtractor()
        result = extractor.fit(X, y)

        assert result is extractor

    def test_transform_output_shape(self, synthetic_eeg_data, n_epochs, n_channels):
        """Test BandPowerExtractor output shape."""
        from features import BandPowerExtractor

        X, y = synthetic_eeg_data
        extractor = BandPowerExtractor()
        X_bp = extractor.fit_transform(X)

        n_bands = len(extractor.freq_bands)
        expected_shape = (n_epochs, n_channels * n_bands)
        assert X_bp.shape == expected_shape

    def test_transform_no_nan(self, synthetic_eeg_data):
        """Test that transform does not produce NaN values."""
        from features import BandPowerExtractor

        X, y = synthetic_eeg_data
        extractor = BandPowerExtractor()
        X_bp = extractor.fit_transform(X)

        assert not np.isnan(X_bp).any()

    def test_transform_positive_values(self, synthetic_eeg_data):
        """Test that band power values are non-negative."""
        from features import BandPowerExtractor

        X, y = synthetic_eeg_data
        extractor = BandPowerExtractor()
        X_bp = extractor.fit_transform(X)

        assert np.all(X_bp >= 0)
