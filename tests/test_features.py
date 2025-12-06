"""
Unit tests for features.py module.

Tests feature extraction classes:
- PSDExtractor
- BandPowerExtractor
- LogVarianceExtractor
- FlattenExtractor
"""

import numpy as np


class TestPSDExtractor:
    """Tests for PSDExtractor class."""

    def test_init_default_params(self):
        """Test PSDExtractor initialization with default parameters."""
        from features import PSDExtractor

        extractor = PSDExtractor()
        assert extractor.fs == 160.0
        assert extractor.nperseg == 256
        assert extractor.noverlap == 128
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


class TestBandPowerExtractor:
    """Tests for BandPowerExtractor class."""

    def test_init_default_params(self):
        """Test BandPowerExtractor initialization with default parameters."""
        from features import BandPowerExtractor

        extractor = BandPowerExtractor()
        assert extractor.fs == 160.0
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


class TestSklearnCompatibility:
    """Tests for sklearn transformer interface compatibility."""

    def test_psd_sklearn_interface(self, synthetic_eeg_data):
        """Test PSDExtractor is sklearn compatible."""
        from features import PSDExtractor

        # Basic interface check - has fit, transform, fit_transform
        extractor = PSDExtractor()
        assert hasattr(extractor, 'fit')
        assert hasattr(extractor, 'transform')
        assert hasattr(extractor, 'fit_transform')
        assert hasattr(extractor, 'get_params')
        assert hasattr(extractor, 'set_params')

    def test_bandpower_sklearn_interface(self, synthetic_eeg_data):
        """Test BandPowerExtractor is sklearn compatible."""
        from features import BandPowerExtractor

        extractor = BandPowerExtractor()
        assert hasattr(extractor, 'fit')
        assert hasattr(extractor, 'transform')
        assert hasattr(extractor, 'fit_transform')
        assert hasattr(extractor, 'get_params')
        assert hasattr(extractor, 'set_params')

    def test_log_variance_sklearn_interface(self):
        """Test LogVarianceExtractor is sklearn compatible."""
        from features import LogVarianceExtractor

        extractor = LogVarianceExtractor()
        assert hasattr(extractor, 'fit')
        assert hasattr(extractor, 'transform')
        assert hasattr(extractor, 'fit_transform')

    def test_flatten_sklearn_interface(self):
        """Test FlattenExtractor is sklearn compatible."""
        from features import FlattenExtractor

        extractor = FlattenExtractor()
        assert hasattr(extractor, 'fit')
        assert hasattr(extractor, 'transform')
        assert hasattr(extractor, 'fit_transform')
