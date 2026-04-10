"""
Prediction module for EEG classification.

Handles:
- Loading trained models
- Real-time prediction simulation
- Batch prediction
"""

import os
import time
import logging
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import display
from constants import MAX_PREDICTION_TIME
from preprocess import preprocess_subject
from training import load_model

# Configure logging
logger = logging.getLogger(__name__)


def predict_single_epoch(pipeline: Pipeline, epoch: NDArray[np.float64]) -> Tuple[int, float]:
    """
    Predict class for a single EEG epoch.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline
    epoch : np.ndarray
        Single epoch of shape (n_channels, n_times)

    Returns
    -------
    prediction : int
        Predicted class
    prediction_time : float
        Time taken for prediction (seconds)
    """
    # Ensure epoch has batch dimension
    if epoch.ndim == 2:
        epoch = epoch[np.newaxis, ...]

    start_time = time.time()
    prediction = pipeline.predict(epoch)
    prediction_time = time.time() - start_time

    return prediction[0], prediction_time


def predict_batch(pipeline: Pipeline, X: NDArray[np.float64]) -> Tuple[NDArray[np.int64], float]:
    """
    Predict classes for a batch of epochs.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline
    X : np.ndarray
        Batch of epochs of shape (n_epochs, n_channels, n_times)

    Returns
    -------
    predictions : np.ndarray
        Predicted classes
    total_time : float
        Total time for predictions
    """
    start_time = time.time()
    predictions = pipeline.predict(X)
    total_time = time.time() - start_time

    return predictions, total_time


def simulate_realtime_prediction(
        pipeline: Pipeline, X: NDArray[np.float64], y: NDArray[np.int64],
        delay: float = 0.0,
        max_time: float = MAX_PREDICTION_TIME,
        verbose: bool = True) -> Dict[str, Any]:
    """
    Simulate real-time prediction of EEG epochs.

    Processes epochs one by one, simulating a streaming scenario.
    Each prediction must complete within max_time seconds.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline
    X : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times)
    y : np.ndarray
        True labels
    delay : float
        Artificial delay between epochs (for visualization)
    max_time : float
        Maximum allowed time per prediction (seconds)
    verbose : bool
        Whether to print per-epoch results

    Returns
    -------
    results : dict
        Dictionary containing:
        - predictions: array of predictions
        - true_labels: array of true labels
        - times: array of prediction times
        - accuracy: overall accuracy
        - within_time_limit: whether all predictions were within time limit
    """
    n_epochs = len(y)
    predictions = np.zeros(n_epochs, dtype=int)
    times = np.zeros(n_epochs)
    correct = np.zeros(n_epochs, dtype=bool)

    if verbose:
        display.print_realtime_header(n_epochs, max_time)

    for i in range(n_epochs):
        pred, pred_time = predict_single_epoch(pipeline, X[i])

        predictions[i] = pred
        times[i] = pred_time
        correct[i] = (pred == y[i])

        if verbose:
            display.print_realtime_epoch(i, int(pred), int(y[i]), pred_time, max_time)

        if delay > 0:
            time.sleep(delay)

    accuracy = accuracy_score(y, predictions)
    avg_time = np.mean(times)
    max_pred_time: float = np.max(times)
    within_limit = bool(max_pred_time < max_time)

    if verbose:
        display.print_realtime_summary(accuracy, n_epochs, avg_time, max_pred_time,
                                       within_limit, max_time)

    return {
        'predictions': predictions,
        'true_labels': y,
        'times': times,
        'accuracy': accuracy,
        'avg_time': avg_time,
        'max_time': max_pred_time,
        'within_time_limit': within_limit
    }


def run_prediction(subject: int, runs: List[int],
                   model_path: Optional[str] = None,
                   model_dir: str = 'models',
                   pipeline_name: str = 'csp_lda',
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Run prediction on a subject's data.

    Parameters
    ----------
    subject : int
        Subject number
    runs : list
        List of run numbers
    model_path : str
        Path to the saved model (if None, constructs from subject/run)
    model_dir : str
        Directory containing saved models
    pipeline_name : str
        Name of the pipeline (for model path construction)
    verbose : bool
        Whether to print results

    Returns
    -------
    results : dict
        Prediction results
    """
    # Construct model path if not provided
    if model_path is None:
        model_path = os.path.join(
            model_dir,
            f"model_s{subject}_r{runs[0]}_{pipeline_name}.pkl"
        )

    if verbose:
        display.section(f"BCI Prediction — Subject: {subject}, Runs: {runs}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if verbose:
        print("\nLoading model...")

    pipeline, metadata = load_model(model_path)

    if verbose:
        display.print_model_info(model_path, metadata)
        print("\nLoading test data...")

    X, y, _ = preprocess_subject(subject, runs)

    if verbose:
        display.print_data_info(X, y)

    # Run real-time simulation
    results = simulate_realtime_prediction(pipeline, X, y, verbose=verbose)

    return results


def predict_from_file(model_path: str, data_path: str,
                      verbose: bool = True) -> Dict[str, Any]:
    """
    Predict using saved model on data from file.

    Parameters
    ----------
    model_path : str
        Path to the saved model
    data_path : str
        Path to the data file (npz format with X and y)
    verbose : bool
        Whether to print results

    Returns
    -------
    results : dict
        Prediction results
    """
    # Load model
    pipeline, metadata = load_model(model_path)

    # Load data
    data = np.load(data_path)
    X = data['X']
    y = data['y']

    # Run prediction
    results = simulate_realtime_prediction(pipeline, X, y, verbose=verbose)

    return results


if __name__ == "__main__":
    import sys

    # Default: predict on subject 1, run 6
    subject = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    display.section("EEG BCI Prediction")
    results = run_prediction(subject, [run])
    print(f"\nFinal accuracy: {results['accuracy']:.4f}")
