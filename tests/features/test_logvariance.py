"""
Unit tests for LogVarianceExtractor class.
"""

import numpy as np


class TestLogVarianceExtractor:
    """Tests for LogVarianceExtractor class."""

    def test_init(self):
        """Test LogVarianceExtractor initialization."""
        from features import LogVarianceExtractor

        extractor = LogVarianceExtractor()
        assert extractor is not None

    def test_fit_returns_self(self, synthetic_eeg_data):
        """Test that fit returns self."""
        from features import LogVarianceExtractor

        X, y = synthetic_eeg_data
        extractor = LogVarianceExtractor()
        result = extractor.fit(X, y)

        assert result is extractor

    def test_transform_3d_input(self, synthetic_eeg_data, n_epochs, n_channels):
        """Test LogVarianceExtractor with 3D input."""
        from features import LogVarianceExtractor

        X, y = synthetic_eeg_data
        extractor = LogVarianceExtractor()
        X_lv = extractor.fit_transform(X)

        expected_shape = (n_epochs, n_channels)
        assert X_lv.shape == expected_shape

    def test_transform_2d_input(self, flat_2d_data):
        """Test LogVarianceExtractor with 2D input (passthrough with log)."""
        from features import LogVarianceExtractor

        X, y = flat_2d_data
        extractor = LogVarianceExtractor()
        X_lv = extractor.fit_transform(X)

        # 2D input should apply log transform directly
        expected = np.log(X + 1e-10)
        np.testing.assert_array_almost_equal(X_lv, expected)

    def test_transform_no_nan(self, synthetic_eeg_data):
        """Test that transform does not produce NaN values."""
        from features import LogVarianceExtractor

        X, y = synthetic_eeg_data
        extractor = LogVarianceExtractor()
        X_lv = extractor.fit_transform(X)

        assert not np.isnan(X_lv).any()

    def test_variance_decreases_for_small_amplitude(self):
        """Test that smaller amplitude signals have smaller variance."""
        from features import LogVarianceExtractor

        np.random.seed(42)
        X_large = np.random.randn(10, 8, 100) * 10
        X_small = np.random.randn(10, 8, 100) * 0.1

        extractor = LogVarianceExtractor()
        lv_large = extractor.fit_transform(X_large)
        lv_small = extractor.fit_transform(X_small)

        # Log variance should be smaller for smaller amplitude
        assert np.mean(lv_small) < np.mean(lv_large)
