"""
Unit tests for CSP-based pipelines.
"""

from sklearn.pipeline import Pipeline


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
