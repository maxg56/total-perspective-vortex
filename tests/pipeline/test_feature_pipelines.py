"""
Unit tests for feature extraction pipelines.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sklearn.pipeline import Pipeline
from constants import EEG_SAMPLING_RATE, DEFAULT_N_COMPONENTS_PCA


class TestBuildPsdLdaPipeline:
    """Tests for build_psd_lda_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_psd_lda_pipeline returns Pipeline."""
        from pipeline import build_psd_lda_pipeline

        pipeline = build_psd_lda_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        """Test pipeline has PSD, scaler, and LDA steps."""
        from pipeline import build_psd_lda_pipeline

        pipeline = build_psd_lda_pipeline()
        steps = dict(pipeline.steps)
        assert 'psd' in steps
        assert 'scaler' in steps
        assert 'lda' in steps

    def test_fs_parameter(self):
        """Test sampling frequency parameter."""
        from pipeline import build_psd_lda_pipeline

        pipeline = build_psd_lda_pipeline(fs=250.0)
        psd = dict(pipeline.steps)['psd']
        assert psd.fs == 250.0

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_psd_lda_pipeline

        X, y = small_synthetic_data
        pipeline = build_psd_lda_pipeline()
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)


class TestBuildBandpowerLdaPipeline:
    """Tests for build_bandpower_lda_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_bandpower_lda_pipeline returns Pipeline."""
        from pipeline import build_bandpower_lda_pipeline

        pipeline = build_bandpower_lda_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        """Test pipeline has bandpower, scaler, and LDA steps."""
        from pipeline import build_bandpower_lda_pipeline

        pipeline = build_bandpower_lda_pipeline()
        steps = dict(pipeline.steps)
        assert 'bandpower' in steps
        assert 'scaler' in steps
        assert 'lda' in steps

    def test_fs_parameter(self):
        """Test sampling frequency parameter."""
        from pipeline import build_bandpower_lda_pipeline

        pipeline = build_bandpower_lda_pipeline(fs=250.0)
        bp = dict(pipeline.steps)['bandpower']
        assert bp.fs == 250.0

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_bandpower_lda_pipeline

        X, y = small_synthetic_data
        pipeline = build_bandpower_lda_pipeline()
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)


class TestBuildFlatPcaLdaPipeline:
    """Tests for build_flat_pca_lda_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_flat_pca_lda_pipeline returns Pipeline."""
        from pipeline import build_flat_pca_lda_pipeline

        pipeline = build_flat_pca_lda_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        """Test pipeline has flatten, scaler, PCA, and LDA steps."""
        from pipeline import build_flat_pca_lda_pipeline

        pipeline = build_flat_pca_lda_pipeline()
        steps = dict(pipeline.steps)
        assert 'flatten' in steps
        assert 'scaler' in steps
        assert 'pca' in steps
        assert 'lda' in steps

    def test_n_components_parameter(self):
        """Test PCA n_components parameter."""
        from pipeline import build_flat_pca_lda_pipeline

        pipeline = build_flat_pca_lda_pipeline(n_components=30)
        pca = dict(pipeline.steps)['pca']
        assert pca.n_components == 30

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_flat_pca_lda_pipeline

        X, y = small_synthetic_data
        # Use fewer components for small data
        pipeline = build_flat_pca_lda_pipeline(n_components=DEFAULT_N_COMPONENTS_PCA)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)
