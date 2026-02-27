"""
Unit tests for bonus pipelines (wavelet and custom classifier).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sklearn.pipeline import Pipeline


class TestBuildWaveletLdaPipeline:
    """Tests for build_wavelet_lda_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_wavelet_lda_pipeline returns Pipeline."""
        from pipeline import build_wavelet_lda_pipeline

        pipeline = build_wavelet_lda_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        """Test pipeline has wavelet, scaler, and LDA steps."""
        from pipeline import build_wavelet_lda_pipeline

        pipeline = build_wavelet_lda_pipeline()
        steps = dict(pipeline.steps)
        assert 'wavelet' in steps
        assert 'scaler' in steps
        assert 'lda' in steps

    def test_fs_parameter(self):
        """Test sampling frequency parameter."""
        from pipeline import build_wavelet_lda_pipeline

        pipeline = build_wavelet_lda_pipeline(fs=250.0)
        wavelet = dict(pipeline.steps)['wavelet']
        assert wavelet.fs == 250.0

    def test_n_scales_parameter(self):
        """Test n_scales_per_band parameter."""
        from pipeline import build_wavelet_lda_pipeline

        pipeline = build_wavelet_lda_pipeline(n_scales_per_band=10)
        wavelet = dict(pipeline.steps)['wavelet']
        assert wavelet.n_scales_per_band == 10

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_wavelet_lda_pipeline

        X, y = small_synthetic_data
        pipeline = build_wavelet_lda_pipeline()
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))


class TestBuildWaveletCustomPipeline:
    """Tests for build_wavelet_custom_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_wavelet_custom_pipeline returns Pipeline."""
        from pipeline import build_wavelet_custom_pipeline

        pipeline = build_wavelet_custom_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        """Test pipeline has wavelet, scaler, and custom classifier steps."""
        from pipeline import build_wavelet_custom_pipeline

        pipeline = build_wavelet_custom_pipeline()
        steps = dict(pipeline.steps)
        assert 'wavelet' in steps
        assert 'scaler' in steps
        assert 'clf' in steps

    def test_shrinkage_parameter(self):
        """Test shrinkage parameter passes through."""
        from pipeline import build_wavelet_custom_pipeline

        pipeline = build_wavelet_custom_pipeline(shrink_threshold=0.5)
        clf = dict(pipeline.steps)['clf']
        assert clf.shrink_threshold == 0.5

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_wavelet_custom_pipeline

        X, y = small_synthetic_data
        pipeline = build_wavelet_custom_pipeline()
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))


class TestBuildCspCustomPipeline:
    """Tests for build_csp_custom_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_csp_custom_pipeline returns Pipeline."""
        from pipeline import build_csp_custom_pipeline

        pipeline = build_csp_custom_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        """Test pipeline has CSP, scaler, and custom classifier steps."""
        from pipeline import build_csp_custom_pipeline

        pipeline = build_csp_custom_pipeline()
        steps = dict(pipeline.steps)
        assert 'csp' in steps
        assert 'scaler' in steps
        assert 'clf' in steps

    def test_n_components_parameter(self):
        """Test n_components passes through to CSP."""
        from pipeline import build_csp_custom_pipeline

        pipeline = build_csp_custom_pipeline(n_components=8)
        csp = dict(pipeline.steps)['csp']
        assert csp.n_components == 8

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_csp_custom_pipeline

        X, y = small_synthetic_data
        pipeline = build_csp_custom_pipeline(n_components=4)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))


class TestNewPipelinesInRegistry:
    """Tests for new pipelines in the registry."""

    def test_wavelet_lda_in_list(self):
        """Test wavelet_lda is in pipeline list."""
        from pipeline import list_pipelines

        assert 'wavelet_lda' in list_pipelines()

    def test_wavelet_custom_in_list(self):
        """Test wavelet_custom is in pipeline list."""
        from pipeline import list_pipelines

        assert 'wavelet_custom' in list_pipelines()

    def test_csp_custom_in_list(self):
        """Test csp_custom is in pipeline list."""
        from pipeline import list_pipelines

        assert 'csp_custom' in list_pipelines()

    def test_get_pipeline_wavelet_lda(self):
        """Test get_pipeline returns wavelet_lda."""
        from pipeline import get_pipeline

        pipeline = get_pipeline('wavelet_lda')
        assert isinstance(pipeline, Pipeline)

    def test_get_pipeline_wavelet_custom(self):
        """Test get_pipeline returns wavelet_custom."""
        from pipeline import get_pipeline

        pipeline = get_pipeline('wavelet_custom')
        assert isinstance(pipeline, Pipeline)

    def test_get_pipeline_csp_custom(self):
        """Test get_pipeline returns csp_custom."""
        from pipeline import get_pipeline

        pipeline = get_pipeline('csp_custom')
        assert isinstance(pipeline, Pipeline)

    def test_total_pipeline_count(self):
        """Test that we now have 10 pipelines."""
        from pipeline import list_pipelines

        assert len(list_pipelines()) == 10
