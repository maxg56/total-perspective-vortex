"""
Unit tests for pipeline.py module.

Tests pipeline construction functions:
- build_csp_lda_pipeline
- build_csp_svm_pipeline
- build_csp_lr_pipeline
- build_psd_lda_pipeline
- build_bandpower_lda_pipeline
- build_flat_pca_lda_pipeline
- get_pipeline
- list_pipelines
"""

import pytest
import numpy as np
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


class TestBuildCspLdaPipeline:
    """Tests for build_csp_lda_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_csp_lda_pipeline returns Pipeline."""
        from pipeline import build_csp_lda_pipeline

        pipeline = build_csp_lda_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_csp_step(self):
        """Test pipeline has CSP step."""
        from pipeline import build_csp_lda_pipeline
        from mycsp import MyCSP

        pipeline = build_csp_lda_pipeline()
        steps = dict(pipeline.steps)
        assert 'csp' in steps
        assert isinstance(steps['csp'], MyCSP)

    def test_has_lda_step(self):
        """Test pipeline has LDA step."""
        from pipeline import build_csp_lda_pipeline
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        pipeline = build_csp_lda_pipeline()
        steps = dict(pipeline.steps)
        assert 'lda' in steps
        assert isinstance(steps['lda'], LinearDiscriminantAnalysis)

    def test_n_components_parameter(self):
        """Test n_components parameter."""
        from pipeline import build_csp_lda_pipeline

        pipeline = build_csp_lda_pipeline(n_components=8)
        csp = dict(pipeline.steps)['csp']
        assert csp.n_components == 8

    def test_reg_parameter(self):
        """Test regularization parameter."""
        from pipeline import build_csp_lda_pipeline

        pipeline = build_csp_lda_pipeline(reg=0.1)
        csp = dict(pipeline.steps)['csp']
        assert csp.reg == 0.1

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_csp_lda_pipeline

        X, y = small_synthetic_data
        pipeline = build_csp_lda_pipeline(n_components=4)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))


class TestBuildCspSvmPipeline:
    """Tests for build_csp_svm_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_csp_svm_pipeline returns Pipeline."""
        from pipeline import build_csp_svm_pipeline

        pipeline = build_csp_svm_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        """Test pipeline has CSP, scaler, and SVM steps."""
        from pipeline import build_csp_svm_pipeline

        pipeline = build_csp_svm_pipeline()
        steps = dict(pipeline.steps)
        assert 'csp' in steps
        assert 'scaler' in steps
        assert 'svm' in steps

    def test_svm_parameters(self):
        """Test SVM parameters can be customized."""
        from pipeline import build_csp_svm_pipeline

        pipeline = build_csp_svm_pipeline(C=0.5, kernel='linear')
        svm = dict(pipeline.steps)['svm']
        assert svm.C == 0.5
        assert svm.kernel == 'linear'

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_csp_svm_pipeline

        X, y = small_synthetic_data
        pipeline = build_csp_svm_pipeline(n_components=4)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)


class TestBuildCspLrPipeline:
    """Tests for build_csp_lr_pipeline function."""

    def test_returns_pipeline(self):
        """Test that build_csp_lr_pipeline returns Pipeline."""
        from pipeline import build_csp_lr_pipeline

        pipeline = build_csp_lr_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        """Test pipeline has CSP, scaler, and LR steps."""
        from pipeline import build_csp_lr_pipeline

        pipeline = build_csp_lr_pipeline()
        steps = dict(pipeline.steps)
        assert 'csp' in steps
        assert 'scaler' in steps
        assert 'lr' in steps

    def test_lr_parameters(self):
        """Test LR parameters can be customized."""
        from pipeline import build_csp_lr_pipeline

        pipeline = build_csp_lr_pipeline(C=0.5)
        lr = dict(pipeline.steps)['lr']
        assert lr.C == 0.5

    def test_fit_predict(self, small_synthetic_data):
        """Test pipeline can fit and predict."""
        from pipeline import build_csp_lr_pipeline

        X, y = small_synthetic_data
        pipeline = build_csp_lr_pipeline(n_components=4)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)


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
        pipeline = build_flat_pca_lda_pipeline(n_components=10)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)


class TestPipelineIntegration:
    """Integration tests for all pipelines."""

    def test_all_pipelines_fit_predict(self, small_synthetic_data):
        """Test that all pipelines can fit and predict."""
        from pipeline import get_pipeline, list_pipelines

        X, y = small_synthetic_data

        for name in list_pipelines():
            # Adjust parameters for small data
            if name.startswith('csp_'):
                pipeline = get_pipeline(name, n_components=4)
            elif name == 'flat_pca_lda':
                pipeline = get_pipeline(name, n_components=10)
            else:
                pipeline = get_pipeline(name)

            pipeline.fit(X, y)
            predictions = pipeline.predict(X)

            assert len(predictions) == len(y), f"Pipeline {name} failed"

    def test_pipeline_returns_valid_predictions(self, small_synthetic_data):
        """Test that predictions are valid class labels."""
        from pipeline import get_pipeline

        X, y = small_synthetic_data
        unique_labels = set(np.unique(y))

        pipeline = get_pipeline('csp_lda', n_components=4)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        for pred in predictions:
            assert pred in unique_labels
