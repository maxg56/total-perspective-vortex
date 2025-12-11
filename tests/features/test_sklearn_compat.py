"""
Tests for sklearn transformer interface compatibility.
"""


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
