"""
Unit tests for pipeline factory functions.
"""

import pytest
from sklearn.pipeline import Pipeline


class TestListPipelines:
    """Tests for list_pipelines function."""

    def test_returns_list(self):
        """Test that list_pipelines returns a list."""
        from pipeline import list_pipelines

        result = list_pipelines()
        assert isinstance(result, list)

    def test_contains_expected_pipelines(self):
        """Test that all expected pipelines are listed."""
        from pipeline import list_pipelines

        result = list_pipelines()
        expected = ['csp_lda', 'csp_svm', 'csp_lr', 'psd_lda', 'bandpower_lda', 'flat_pca_lda']

        for name in expected:
            assert name in result

    def test_returns_six_pipelines(self):
        """Test that exactly 6 pipelines are available."""
        from pipeline import list_pipelines

        result = list_pipelines()
        assert len(result) == 6


class TestGetPipeline:
    """Tests for get_pipeline factory function."""

    def test_get_pipeline_returns_pipeline(self):
        """Test that get_pipeline returns sklearn Pipeline."""
        from pipeline import get_pipeline

        for name in ['csp_lda', 'csp_svm', 'csp_lr', 'psd_lda', 'bandpower_lda', 'flat_pca_lda']:
            pipeline = get_pipeline(name)
            assert isinstance(pipeline, Pipeline)

    def test_get_pipeline_default_is_csp_lda(self):
        """Test that default pipeline is csp_lda."""
        from pipeline import get_pipeline

        pipeline = get_pipeline()
        steps = dict(pipeline.steps)
        assert 'csp' in steps
        assert 'lda' in steps

    def test_get_pipeline_invalid_name_raises_error(self):
        """Test that invalid pipeline name raises ValueError."""
        from pipeline import get_pipeline

        with pytest.raises(ValueError, match="Unknown pipeline"):
            get_pipeline('invalid_pipeline_name')

    def test_get_pipeline_passes_kwargs(self):
        """Test that kwargs are passed to pipeline builders."""
        from pipeline import get_pipeline

        pipeline = get_pipeline('csp_lda', n_components=8)
        csp = dict(pipeline.steps)['csp']
        assert csp.n_components == 8
