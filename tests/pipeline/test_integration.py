"""
Integration tests for all pipelines.
"""

import numpy as np


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
