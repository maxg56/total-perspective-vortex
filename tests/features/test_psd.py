"""
Unit tests for PSDExtractor class.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from constants import EEG_SAMPLING_RATE, WELCH_NPERSEG, WELCH_NOVERLAP


class TestPSDExtractor:
    """Tests for PSDExtractor class."""

    def test_init_default_params(self):
        """Test PSDExtractor initialization with default parameters."""
        from features import PSDExtractor

        extractor = PSDExtractor()
        assert extractor.fs == EEG_SAMPLING_RATE
        assert extractor.nperseg == WELCH_NPERSEG
        assert extractor.noverlap == WELCH_NOVERLAP
        assert 'mu' in extractor.freq_bands
        assert 'beta' in extractor.freq_bands

    def test_init_custom_params(self):
        """Test PSDExtractor with custom parameters."""
        from features import PSDExtractor

        custom_bands = {'alpha': (8, 13), 'beta': (13, 30)}
        extractor = PSDExtractor(fs=250.0, nperseg=512, noverlap=256, freq_bands=custom_bands)

        assert extractor.fs == 250.0
        assert extractor.nperseg == 512
        assert extractor.noverlap == 256
        assert extractor.freq_bands == custom_bands

    def test_fit_returns_self(self, synthetic_eeg_data):
        """Test that fit returns self."""
        from features import PSDExtractor

        X, y = synthetic_eeg_data
        extractor = PSDExtractor()
        result = extractor.fit(X, y)

        assert result is extractor

    def test_transform_output_shape(self, synthetic_eeg_data, n_epochs, n_channels):
        """Test PSDExtractor output shape."""
        from features import PSDExtractor

        X, y = synthetic_eeg_data
        extractor = PSDExtractor()
        X_psd = extractor.fit_transform(X)

        n_bands = len(extractor.freq_bands)
        expected_shape = (n_epochs, n_channels * n_bands)
        assert X_psd.shape == expected_shape

    def test_transform_no_nan(self, synthetic_eeg_data):
        """Test that transform does not produce NaN values."""
        from features import PSDExtractor

        X, y = synthetic_eeg_data
        extractor = PSDExtractor()
        X_psd = extractor.fit_transform(X)

        assert not np.isnan(X_psd).any()

    def test_transform_positive_values(self, synthetic_eeg_data):
        """Test that PSD values are non-negative."""
        from features import PSDExtractor

        X, y = synthetic_eeg_data
        extractor = PSDExtractor()
        X_psd = extractor.fit_transform(X)

        assert np.all(X_psd >= 0)

    def test_small_nperseg_handling(self, small_synthetic_data):
        """Test handling when nperseg > n_times."""
        from features import PSDExtractor

        X, y = small_synthetic_data
        # nperseg larger than n_times
        extractor = PSDExtractor(nperseg=1024)
        X_psd = extractor.fit_transform(X)

        assert X_psd.shape[0] == X.shape[0]
        assert not np.isnan(X_psd).any()
