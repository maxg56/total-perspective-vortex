"""
Unit tests for train.py module.

Tests training and model persistence functions:
- train_and_evaluate
- train_with_holdout
- compare_pipelines
- save_model
- load_model
"""

import pytest
import numpy as np
import os


class TestTrainAndEvaluate:
    """Tests for train_and_evaluate function."""

    def test_returns_tuple(self, small_synthetic_data):
        """Test that train_and_evaluate returns tuple of (pipeline, scores)."""
        from train import train_and_evaluate

        X, y = small_synthetic_data
        result = train_and_evaluate(X, y, pipeline_name='csp_lda', cv=3, verbose=False, n_components=4)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_fitted_pipeline(self, small_synthetic_data):
        """Test that returned pipeline is fitted."""
        from train import train_and_evaluate

        X, y = small_synthetic_data
        pipeline, scores = train_and_evaluate(X, y, pipeline_name='csp_lda', cv=3, verbose=False, n_components=4)

        # Pipeline should be fitted and able to predict
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_returns_cv_scores(self, small_synthetic_data):
        """Test that CV scores are returned correctly."""
        from train import train_and_evaluate

        X, y = small_synthetic_data
        cv = 3
        pipeline, scores = train_and_evaluate(X, y, pipeline_name='csp_lda', cv=cv, verbose=False, n_components=4)

        assert len(scores) == cv
        assert all(0 <= s <= 1 for s in scores)

    def test_supports_different_pipelines(self, small_synthetic_data):
        """Test training with different pipeline types."""
        from train import train_and_evaluate

        X, y = small_synthetic_data

        # Test CSP + LDA
        pipeline1, scores1 = train_and_evaluate(X, y, pipeline_name='csp_lda', cv=3, verbose=False, n_components=4)
        assert len(scores1) == 3

        # Test PSD + LDA
        pipeline2, scores2 = train_and_evaluate(X, y, pipeline_name='psd_lda', cv=3, verbose=False)
        assert len(scores2) == 3

    def test_verbose_output(self, small_synthetic_data, capsys):
        """Test verbose mode produces output."""
        from train import train_and_evaluate

        X, y = small_synthetic_data
        train_and_evaluate(X, y, pipeline_name='csp_lda', cv=3, verbose=True, n_components=4)

        captured = capsys.readouterr()
        assert 'Pipeline: csp_lda' in captured.out
        assert 'Cross-validation scores' in captured.out


class TestTrainWithHoldout:
    """Tests for train_with_holdout function."""

    def test_returns_tuple(self, small_synthetic_data):
        """Test that train_with_holdout returns tuple of (pipeline, cv_scores, test_accuracy)."""
        from train import train_with_holdout

        X, y = small_synthetic_data
        result = train_with_holdout(X, y, pipeline_name='csp_lda', test_size=0.3, cv=2, verbose=False, n_components=4)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_fitted_pipeline(self, small_synthetic_data):
        """Test that returned pipeline is fitted."""
        from train import train_with_holdout

        X, y = small_synthetic_data
        pipeline, cv_scores, test_accuracy = train_with_holdout(
            X, y, pipeline_name='csp_lda', test_size=0.3, cv=2, verbose=False, n_components=4
        )

        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_test_accuracy_valid(self, small_synthetic_data):
        """Test that test accuracy is valid."""
        from train import train_with_holdout

        X, y = small_synthetic_data
        pipeline, cv_scores, test_accuracy = train_with_holdout(
            X, y, pipeline_name='csp_lda', test_size=0.3, cv=2, verbose=False, n_components=4
        )

        assert 0 <= test_accuracy <= 1

    def test_stratified_split(self, small_synthetic_data):
        """Test that train/test split is stratified."""
        from train import train_with_holdout
        from sklearn.model_selection import train_test_split

        X, y = small_synthetic_data
        # Just verify no error occurs with stratification
        train_with_holdout(X, y, pipeline_name='csp_lda', test_size=0.3, cv=2, verbose=False, n_components=4)


class TestComparePipelines:
    """Tests for compare_pipelines function."""

    def test_returns_dict(self, small_synthetic_data):
        """Test that compare_pipelines returns dictionary."""
        from train import compare_pipelines

        X, y = small_synthetic_data
        result = compare_pipelines(X, y, cv=2, verbose=False)

        assert isinstance(result, dict)

    def test_contains_all_pipelines(self, small_synthetic_data):
        """Test that result contains all pipeline names."""
        from train import compare_pipelines
        from pipeline import list_pipelines

        X, y = small_synthetic_data
        result = compare_pipelines(X, y, cv=2, verbose=False)

        for name in list_pipelines():
            assert name in result

    def test_result_structure(self, small_synthetic_data):
        """Test result dictionary structure."""
        from train import compare_pipelines

        X, y = small_synthetic_data
        result = compare_pipelines(X, y, cv=2, verbose=False)

        for name, data in result.items():
            if data is not None:  # Some might fail
                assert 'mean' in data
                assert 'std' in data
                assert 'scores' in data
                assert 0 <= data['mean'] <= 1

    def test_verbose_output(self, small_synthetic_data, capsys):
        """Test verbose mode produces output."""
        from train import compare_pipelines

        X, y = small_synthetic_data
        compare_pipelines(X, y, cv=2, verbose=True)

        captured = capsys.readouterr()
        assert 'Comparing all pipelines' in captured.out
        assert 'Best pipeline' in captured.out


class TestSaveModel:
    """Tests for save_model function."""

    def test_save_creates_file(self, trained_csp_pipeline, temp_model_dir):
        """Test that save_model creates a file."""
        from train import save_model

        pipeline, X, y = trained_csp_pipeline
        path = os.path.join(temp_model_dir, 'test_model.pkl')
        save_model(pipeline, path)

        assert os.path.exists(path)

    def test_save_with_metadata(self, trained_csp_pipeline, temp_model_dir):
        """Test saving model with metadata."""
        from train import save_model, load_model

        pipeline, X, y = trained_csp_pipeline
        metadata = {'subject': 1, 'runs': [6], 'accuracy': 0.85}
        path = os.path.join(temp_model_dir, 'test_model.pkl')

        save_model(pipeline, path, metadata)

        loaded_pipeline, loaded_metadata = load_model(path)
        assert loaded_metadata == metadata

    def test_save_creates_directory(self, trained_csp_pipeline, tmp_path):
        """Test that save_model creates directory if needed."""
        from train import save_model

        pipeline, X, y = trained_csp_pipeline
        path = os.path.join(str(tmp_path), 'new_dir', 'model.pkl')

        save_model(pipeline, path)
        assert os.path.exists(path)


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_returns_tuple(self, trained_csp_pipeline, temp_model_dir):
        """Test that load_model returns tuple of (pipeline, metadata)."""
        from train import save_model, load_model

        pipeline, X, y = trained_csp_pipeline
        path = os.path.join(temp_model_dir, 'test_model.pkl')
        save_model(pipeline, path, {'test': 'data'})

        result = load_model(path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loaded_pipeline_works(self, trained_csp_pipeline, temp_model_dir):
        """Test that loaded pipeline can make predictions."""
        from train import save_model, load_model

        pipeline, X, y = trained_csp_pipeline
        path = os.path.join(temp_model_dir, 'test_model.pkl')
        save_model(pipeline, path)

        loaded_pipeline, _ = load_model(path)
        predictions = loaded_pipeline.predict(X)

        assert len(predictions) == len(y)

    def test_loaded_predictions_match_original(self, trained_csp_pipeline, temp_model_dir):
        """Test that loaded pipeline gives same predictions as original."""
        from train import save_model, load_model

        pipeline, X, y = trained_csp_pipeline
        original_predictions = pipeline.predict(X)

        path = os.path.join(temp_model_dir, 'test_model.pkl')
        save_model(pipeline, path)

        loaded_pipeline, _ = load_model(path)
        loaded_predictions = loaded_pipeline.predict(X)

        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_load_nonexistent_file_raises_error(self, temp_model_dir):
        """Test that loading nonexistent file raises error."""
        from train import load_model

        with pytest.raises(FileNotFoundError):
            load_model(os.path.join(temp_model_dir, 'nonexistent.pkl'))


class TestModelPersistence:
    """Tests for complete model persistence workflow."""

    def test_save_load_roundtrip(self, small_synthetic_data, temp_model_dir):
        """Test complete save/load roundtrip."""
        from train import train_and_evaluate, save_model, load_model

        X, y = small_synthetic_data

        # Train model
        pipeline, scores = train_and_evaluate(X, y, pipeline_name='csp_lda', cv=3, verbose=False, n_components=4)
        original_predictions = pipeline.predict(X)

        # Save model
        metadata = {
            'cv_scores': scores.tolist(),
            'cv_mean': float(scores.mean())
        }
        path = os.path.join(temp_model_dir, 'roundtrip_model.pkl')
        save_model(pipeline, path, metadata)

        # Load model
        loaded_pipeline, loaded_metadata = load_model(path)
        loaded_predictions = loaded_pipeline.predict(X)

        # Verify
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        assert loaded_metadata['cv_mean'] == float(scores.mean())
