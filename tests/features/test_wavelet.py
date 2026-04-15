"""
Unit tests for WaveletExtractor class.
"""

import numpy as np

from constants import EEG_SAMPLING_RATE


class TestWaveletExtractorInit:
    """Tests for WaveletExtractor initialization."""

    def test_init_default_params(self):
        """Test WaveletExtractor initialization with default parameters."""
        from features import WaveletExtractor

        extractor = WaveletExtractor()
        assert extractor.fs == EEG_SAMPLING_RATE
        assert 'mu' in extractor.freq_bands
        assert 'beta' in extractor.freq_bands
        assert extractor.n_scales_per_band == 5
        assert extractor.omega0 == 5.0

    def test_init_custom_params(self):
        """Test WaveletExtractor with custom parameters."""
        from features import WaveletExtractor

        custom_bands = {'alpha': (8, 13)}
        extractor = WaveletExtractor(
            fs=250.0, freq_bands=custom_bands,
            n_scales_per_band=10, omega0=6.0)

        assert extractor.fs == 250.0
        assert extractor.freq_bands == custom_bands
        assert extractor.n_scales_per_band == 10
        assert extractor.omega0 == 6.0


class TestWaveletExtractorFit:
    """Tests for WaveletExtractor.fit method."""

    def test_fit_returns_self(self, synthetic_eeg_data):
        """Test that fit returns self."""
        from features import WaveletExtractor

        X, y = synthetic_eeg_data
        extractor = WaveletExtractor()
        result = extractor.fit(X, y)
        assert result is extractor


class TestWaveletExtractorTransform:
    """Tests for WaveletExtractor.transform method."""

    def test_transform_output_shape(self, synthetic_eeg_data, n_epochs,
                                    n_channels):
        """Test WaveletExtractor output shape."""
        from features import WaveletExtractor

        X, y = synthetic_eeg_data
        extractor = WaveletExtractor()
        X_wav = extractor.fit_transform(X)

        n_bands = len(extractor.freq_bands)
        expected_shape = (n_epochs, n_channels * n_bands)
        assert X_wav.shape == expected_shape

    def test_transform_no_nan(self, synthetic_eeg_data):
        """Test that transform does not produce NaN values."""
        from features import WaveletExtractor

        X, y = synthetic_eeg_data
        extractor = WaveletExtractor()
        X_wav = extractor.fit_transform(X)
        assert not np.isnan(X_wav).any()

    def test_transform_positive_values(self, synthetic_eeg_data):
        """Test that wavelet energy values are non-negative."""
        from features import WaveletExtractor

        X, y = synthetic_eeg_data
        extractor = WaveletExtractor()
        X_wav = extractor.fit_transform(X)
        assert np.all(X_wav >= 0)

    def test_transform_custom_bands(self, small_synthetic_data):
        """Test with custom frequency bands."""
        from features import WaveletExtractor

        X, y = small_synthetic_data
        custom_bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
        extractor = WaveletExtractor(freq_bands=custom_bands)
        X_wav = extractor.fit_transform(X)

        n_channels = X.shape[1]
        assert X_wav.shape == (X.shape[0], n_channels * 3)

    def test_transform_different_scales(self, small_synthetic_data):
        """Test with different number of scales per band."""
        from features import WaveletExtractor

        X, y = small_synthetic_data
        extractor = WaveletExtractor(n_scales_per_band=3)
        X_wav = extractor.fit_transform(X)

        assert X_wav.shape[0] == X.shape[0]
        assert not np.isnan(X_wav).any()


class TestWaveletExtractorSklearnCompat:
    """Tests for WaveletExtractor sklearn compatibility."""

    def test_sklearn_interface(self):
        """Test WaveletExtractor has sklearn interface methods."""
        from features import WaveletExtractor

        extractor = WaveletExtractor()
        assert hasattr(extractor, 'fit')
        assert hasattr(extractor, 'transform')
        assert hasattr(extractor, 'fit_transform')
        assert hasattr(extractor, 'get_params')
        assert hasattr(extractor, 'set_params')

    def test_get_set_params(self):
        """Test get_params and set_params."""
        from features import WaveletExtractor

        extractor = WaveletExtractor(fs=160.0, n_scales_per_band=5)
        params = extractor.get_params()

        assert params['fs'] == 160.0
        assert params['n_scales_per_band'] == 5

        extractor.set_params(n_scales_per_band=10)
        assert extractor.n_scales_per_band == 10


class TestMorletWavelet:
    """Tests for the Morlet wavelet implementation."""

    def test_morlet_shape(self):
        """Test Morlet wavelet output shape."""
        from features.wavelet import _morlet_wavelet

        wavelet = _morlet_wavelet(n_points=256, scale=5.0, fs=160.0)
        assert wavelet.shape == (256,)

    def test_morlet_is_complex(self):
        """Test that Morlet wavelet is complex-valued."""
        from features.wavelet import _morlet_wavelet

        wavelet = _morlet_wavelet(n_points=256, scale=5.0, fs=160.0)
        assert np.iscomplexobj(wavelet)

    def test_morlet_finite(self):
        """Test that Morlet wavelet has no NaN or Inf values."""
        from features.wavelet import _morlet_wavelet

        wavelet = _morlet_wavelet(n_points=256, scale=5.0, fs=160.0)
        assert np.all(np.isfinite(wavelet))


class TestCWT:
    """Tests for the custom CWT implementation."""

    def test_cwt_output_shape(self):
        """Test CWT output shape."""
        from features.wavelet import _cwt_morlet

        x = np.random.randn(160)
        scales = np.array([2.0, 4.0, 8.0])
        coeffs = _cwt_morlet(x, scales, fs=160.0)
        assert coeffs.shape == (3, 160)

    def test_cwt_positive_magnitude(self):
        """Test that CWT magnitude coefficients are non-negative."""
        from features.wavelet import _cwt_morlet

        x = np.random.randn(160)
        scales = np.array([2.0, 4.0, 8.0])
        coeffs = _cwt_morlet(x, scales, fs=160.0)
        assert np.all(coeffs >= 0)

    def test_cwt_no_nan(self):
        """Test that CWT produces no NaN values."""
        from features.wavelet import _cwt_morlet

        x = np.random.randn(160)
        scales = np.array([2.0, 4.0, 8.0])
        coeffs = _cwt_morlet(x, scales, fs=160.0)
        assert not np.isnan(coeffs).any()

    def test_cwt_sinusoidal_signal(self):
        """Test CWT detects known frequency in sinusoidal signal."""
        from features.wavelet import _cwt_morlet, _freq_to_scale

        fs = 160.0
        t = np.arange(0, 1.0, 1.0 / fs)
        freq_target = 10.0  # 10 Hz signal
        x = np.sin(2 * np.pi * freq_target * t)

        # Scales for frequencies around 10 Hz and 30 Hz
        scale_10 = _freq_to_scale(10.0, fs)
        scale_30 = _freq_to_scale(30.0, fs)
        scales = np.array([scale_10, scale_30])

        coeffs = _cwt_morlet(x, scales, fs=fs)

        # Energy at 10 Hz should be higher than at 30 Hz
        energy_10 = np.mean(coeffs[0, :] ** 2)
        energy_30 = np.mean(coeffs[1, :] ** 2)
        assert energy_10 > energy_30
