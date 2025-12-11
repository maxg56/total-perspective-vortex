"""
Unit tests for FlattenExtractor class.
"""

import numpy as np


class TestFlattenExtractor:
    """Tests for FlattenExtractor class."""

    def test_init(self):
        """Test FlattenExtractor initialization."""
        from features import FlattenExtractor

        extractor = FlattenExtractor()
        assert extractor is not None

    def test_fit_returns_self(self, synthetic_eeg_data):
        """Test that fit returns self."""
        from features import FlattenExtractor

        X, y = synthetic_eeg_data
        extractor = FlattenExtractor()
        result = extractor.fit(X, y)

        assert result is extractor

    def test_transform_output_shape(self, synthetic_eeg_data, n_epochs, n_channels, n_times):
        """Test FlattenExtractor output shape."""
        from features import FlattenExtractor

        X, y = synthetic_eeg_data
        extractor = FlattenExtractor()
        X_flat = extractor.fit_transform(X)

        expected_shape = (n_epochs, n_channels * n_times)
        assert X_flat.shape == expected_shape

    def test_transform_preserves_data(self, synthetic_eeg_data):
        """Test that flattening preserves all data."""
        from features import FlattenExtractor

        X, y = synthetic_eeg_data
        extractor = FlattenExtractor()
        X_flat = extractor.fit_transform(X)

        # Reshape back and compare
        X_reshaped = X_flat.reshape(X.shape)
        np.testing.assert_array_equal(X, X_reshaped)

    def test_transform_no_nan(self, synthetic_eeg_data):
        """Test that transform does not produce NaN values."""
        from features import FlattenExtractor

        X, y = synthetic_eeg_data
        extractor = FlattenExtractor()
        X_flat = extractor.fit_transform(X)

        assert not np.isnan(X_flat).any()
