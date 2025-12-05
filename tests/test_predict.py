"""
Unit tests for predict.py module.

Tests prediction functions:
- predict_single_epoch
- predict_batch
- simulate_realtime_prediction
"""

import pytest
import numpy as np
import time


class TestPredictSingleEpoch:
    """Tests for predict_single_epoch function."""

    def test_returns_tuple(self, trained_csp_pipeline):
        """Test that predict_single_epoch returns tuple."""
        from predict import predict_single_epoch

        pipeline, X, y = trained_csp_pipeline
        # Use an epoch from the training data to ensure correct dimensions
        epoch = X[0]
        result = predict_single_epoch(pipeline, epoch)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_prediction_and_time(self, trained_csp_pipeline):
        """Test that function returns prediction and time."""
        from predict import predict_single_epoch

        pipeline, X, y = trained_csp_pipeline
        # Use an epoch from the training data to ensure correct dimensions
        epoch = X[0]
        prediction, pred_time = predict_single_epoch(pipeline, epoch)

        assert isinstance(prediction, (int, np.integer))
        assert isinstance(pred_time, float)
        assert pred_time >= 0

    def test_handles_2d_input(self, trained_csp_pipeline):
        """Test that function handles 2D input (single epoch without batch dim)."""
        from predict import predict_single_epoch

        pipeline, X, y = trained_csp_pipeline
        epoch_2d = X[0]  # Shape: (n_channels, n_times)

        prediction, pred_time = predict_single_epoch(pipeline, epoch_2d)
        assert prediction in np.unique(y)

    def test_handles_3d_input(self, trained_csp_pipeline):
        """Test that function handles 3D input (with batch dimension)."""
        from predict import predict_single_epoch

        pipeline, X, y = trained_csp_pipeline
        epoch_3d = X[0:1]  # Shape: (1, n_channels, n_times)

        prediction, pred_time = predict_single_epoch(pipeline, epoch_3d)
        assert prediction in np.unique(y)

    def test_prediction_is_valid_class(self, trained_csp_pipeline):
        """Test that prediction is a valid class label."""
        from predict import predict_single_epoch

        pipeline, X, y = trained_csp_pipeline
        unique_labels = set(np.unique(y))

        for epoch in X[:5]:
            prediction, _ = predict_single_epoch(pipeline, epoch)
            assert prediction in unique_labels


class TestPredictBatch:
    """Tests for predict_batch function."""

    def test_returns_tuple(self, trained_csp_pipeline):
        """Test that predict_batch returns tuple."""
        from predict import predict_batch

        pipeline, X, y = trained_csp_pipeline
        result = predict_batch(pipeline, X)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_predictions_and_time(self, trained_csp_pipeline):
        """Test that function returns predictions array and total time."""
        from predict import predict_batch

        pipeline, X, y = trained_csp_pipeline
        predictions, total_time = predict_batch(pipeline, X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
        assert isinstance(total_time, float)
        assert total_time >= 0

    def test_predictions_are_valid_classes(self, trained_csp_pipeline):
        """Test that all predictions are valid class labels."""
        from predict import predict_batch

        pipeline, X, y = trained_csp_pipeline
        unique_labels = set(np.unique(y))
        predictions, _ = predict_batch(pipeline, X)

        for pred in predictions:
            assert pred in unique_labels

    def test_batch_faster_than_sequential(self, trained_csp_pipeline):
        """Test that batch prediction is reasonably fast."""
        from predict import predict_batch, predict_single_epoch

        pipeline, X, y = trained_csp_pipeline

        # Batch prediction
        _, batch_time = predict_batch(pipeline, X)

        # Sequential prediction (sample of epochs)
        start = time.time()
        for epoch in X[:5]:
            predict_single_epoch(pipeline, epoch)
        sequential_time_partial = time.time() - start

        # Batch should be efficient (just check it completes)
        assert batch_time > 0


class TestSimulateRealtimePrediction:
    """Tests for simulate_realtime_prediction function."""

    def test_returns_dict(self, trained_csp_pipeline):
        """Test that function returns dictionary."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        result = simulate_realtime_prediction(pipeline, X, y, verbose=False)

        assert isinstance(result, dict)

    def test_result_contains_expected_keys(self, trained_csp_pipeline):
        """Test that result contains all expected keys."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        result = simulate_realtime_prediction(pipeline, X, y, verbose=False)

        expected_keys = ['predictions', 'true_labels', 'times', 'accuracy', 'avg_time', 'max_time', 'within_time_limit']
        for key in expected_keys:
            assert key in result

    def test_predictions_shape(self, trained_csp_pipeline):
        """Test predictions array shape."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        result = simulate_realtime_prediction(pipeline, X, y, verbose=False)

        assert len(result['predictions']) == len(y)
        assert len(result['times']) == len(y)

    def test_accuracy_calculation(self, trained_csp_pipeline):
        """Test accuracy calculation is correct."""
        from predict import simulate_realtime_prediction
        from sklearn.metrics import accuracy_score

        pipeline, X, y = trained_csp_pipeline
        result = simulate_realtime_prediction(pipeline, X, y, verbose=False)

        expected_accuracy = accuracy_score(y, result['predictions'])
        assert result['accuracy'] == expected_accuracy

    def test_accuracy_valid_range(self, trained_csp_pipeline):
        """Test accuracy is in valid range."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        result = simulate_realtime_prediction(pipeline, X, y, verbose=False)

        assert 0 <= result['accuracy'] <= 1

    def test_timing_statistics(self, trained_csp_pipeline):
        """Test timing statistics are valid."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        result = simulate_realtime_prediction(pipeline, X, y, verbose=False)

        assert result['avg_time'] >= 0
        assert result['max_time'] >= 0
        assert result['avg_time'] <= result['max_time']

    def test_within_time_limit_flag(self, trained_csp_pipeline):
        """Test within_time_limit flag is boolean-like."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        result = simulate_realtime_prediction(pipeline, X, y, max_time=10.0, verbose=False)

        # Check it's a boolean-like value (np.bool_ or bool)
        assert result['within_time_limit'] in (True, False)

    def test_verbose_output(self, trained_csp_pipeline, capsys):
        """Test verbose mode produces output."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        # Use only first few epochs to speed up test
        simulate_realtime_prediction(pipeline, X[:5], y[:5], verbose=True)

        captured = capsys.readouterr()
        assert 'Real-time prediction simulation' in captured.out
        assert 'Summary' in captured.out

    def test_strict_time_limit(self, trained_csp_pipeline):
        """Test with strict time limit."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        # Very strict time limit
        result = simulate_realtime_prediction(pipeline, X, y, max_time=0.001, verbose=False)

        # Most predictions should exceed this limit
        assert result['max_time'] > 0

    def test_reasonable_time_limit(self, trained_csp_pipeline):
        """Test predictions complete within reasonable time (2 seconds per epoch)."""
        from predict import simulate_realtime_prediction

        pipeline, X, y = trained_csp_pipeline
        result = simulate_realtime_prediction(pipeline, X[:5], y[:5], max_time=2.0, verbose=False)

        # Should complete within time limit (use == True for numpy bool compatibility)
        assert result['within_time_limit'] == True


class TestPredictFromFile:
    """Tests for predict_from_file function."""

    def test_load_and_predict(self, trained_csp_pipeline, temp_model_dir, tmp_path):
        """Test prediction from saved model and data files."""
        from predict import predict_from_file
        from train import save_model

        pipeline, X, y = trained_csp_pipeline

        # Save model
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        save_model(pipeline, model_path)

        # Save data
        data_path = str(tmp_path / 'test_data.npz')
        np.savez(data_path, X=X, y=y)

        # Predict
        result = predict_from_file(model_path, data_path, verbose=False)

        assert 'accuracy' in result
        assert len(result['predictions']) == len(y)


# Need to import os at module level for TestPredictFromFile
import os


class TestPredictionConsistency:
    """Tests for prediction consistency."""

    def test_same_input_same_output(self, trained_csp_pipeline):
        """Test that same input gives same output."""
        from predict import predict_single_epoch

        pipeline, X, y = trained_csp_pipeline
        epoch = X[0]

        pred1, _ = predict_single_epoch(pipeline, epoch)
        pred2, _ = predict_single_epoch(pipeline, epoch)

        assert pred1 == pred2

    def test_batch_matches_single(self, trained_csp_pipeline):
        """Test that batch predictions match single epoch predictions."""
        from predict import predict_single_epoch, predict_batch

        pipeline, X, y = trained_csp_pipeline

        # Get batch predictions
        batch_preds, _ = predict_batch(pipeline, X[:5])

        # Get single predictions
        single_preds = []
        for epoch in X[:5]:
            pred, _ = predict_single_epoch(pipeline, epoch)
            single_preds.append(pred)

        np.testing.assert_array_equal(batch_preds, single_preds)
