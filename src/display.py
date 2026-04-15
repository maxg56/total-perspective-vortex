"""
display.py - Centralized terminal output functions for the BCI system.

All user-visible text printed to stdout passes through this module,
keeping formatting consistent and easy to update in one place.
"""

from typing import Any, Dict, Optional
import numpy as np
from numpy.typing import NDArray

from constants import SEPARATOR_WIDTH, TARGET_ACCURACY, MAX_PREDICTION_TIME

SEP = '=' * SEPARATOR_WIDTH


def section(title: str) -> None:
    """Print a section header surrounded by separators."""
    print(f"\n{SEP}")
    print(title)
    print(SEP)


def step(message: str) -> None:
    """Print a single-line status step."""
    print(f"\n{message}")


def print_header(subject: int, run: int, mode: str, pipeline: str, run_type: str) -> None:
    """Print the main program header."""
    print(f"\n{SEP}")
    print("    Total Perspective Vortex - EEG BCI System")
    print(SEP)
    print(f"Subject:  {subject}")
    print(f"Run:      {run} ({run_type})")
    print(f"Mode:     {mode}")
    print(f"Pipeline: {pipeline}")
    print(SEP)


def print_data_info(X: NDArray[np.float64], y: NDArray[np.int64]) -> None:
    """Print dataset shape and class distribution."""
    print(f"Data shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")


def print_cv_result(
        scores: NDArray[np.float64],
        pipeline_name: str,
        target: float = TARGET_ACCURACY) -> None:
    """Print cross-validation scores and target comparison."""
    section(f"Pipeline: {pipeline_name}")
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"Min: {scores.min():.4f}, Max: {scores.max():.4f}")
    if scores.mean() >= target:
        print(f"Target accuracy ({target:.0%}) ACHIEVED")
    else:
        gap = target - scores.mean()
        print(f"Target accuracy ({target:.0%}) NOT achieved - need {gap:.4f} more")


def print_pipeline_result(name: str, scores: NDArray[np.float64]) -> None:
    """Print result for one pipeline during comparison."""
    print(f"\n{name}:")
    print(f"  Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")


def print_best_pipeline(name: str, accuracy: float) -> None:
    """Print the winning pipeline from a comparison."""
    print(f"\nBest pipeline: {name} with {accuracy:.4f} accuracy")


def print_training_summary(
        scores: NDArray[np.float64],
        exp_idx: Optional[int],
        exp_target: float) -> None:
    """Print training completion summary with target check."""
    section("TRAINING COMPLETE")
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    if scores.mean() >= exp_target:
        print(f"\n*** TARGET ACCURACY (exp {exp_idx}: {exp_target:.0%}) ACHIEVED! ***")
    else:
        gap = exp_target - scores.mean()
        print(f"\n*** Target accuracy not reached "
              f"(exp {exp_idx}: need {gap:.4f} more for {exp_target:.0%}) ***")


def print_prediction_summary(
        results: Dict[str, Any],
        exp_idx: Optional[int],
        exp_target: float) -> None:
    """Print prediction completion summary with target check."""
    section("PREDICTION COMPLETE")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Average prediction time: {results['avg_time'] * 1000:.2f} ms")
    print(f"Max prediction time: {results['max_time'] * 1000:.2f} ms")
    status = 'PASSED' if results['within_time_limit'] else 'FAILED'
    print(f"Time limit ({MAX_PREDICTION_TIME}s): {status}")
    if results['accuracy'] >= exp_target:
        print(f"\n*** TARGET ACCURACY (exp {exp_idx}: {exp_target:.0%}) ACHIEVED! ***")
    else:
        print(f"\n*** Target accuracy not reached "
              f"(exp {exp_idx}: {exp_target:.0%}) ***")


def print_realtime_header(n_epochs: int, max_time: float = MAX_PREDICTION_TIME) -> None:
    """Print header for real-time prediction simulation."""
    section("Real-time prediction simulation")
    print(f"Processing {n_epochs} epochs (max {max_time}s per epoch)")
    print(SEP)


def print_realtime_epoch(
        i: int, pred: int, true: int,
        elapsed: float, max_time: float = MAX_PREDICTION_TIME) -> None:
    """Print result for a single epoch in real-time simulation."""
    status = "True" if pred == true else "False"
    time_flag = "" if elapsed <= max_time else " [SLOW]"
    print(f"epoch {i:02d}: [{pred}] [{true}] {status}{time_flag}")


def print_realtime_summary(
        accuracy: float, n_epochs: int,
        avg_time: float, max_pred_time: float,
        within_limit: bool,
        max_time: float = MAX_PREDICTION_TIME) -> None:
    """Print summary of the real-time prediction simulation."""
    section("Summary")
    correct = int(accuracy * n_epochs)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{n_epochs})")
    print(f"Average prediction time: {avg_time * 1000:.2f} ms")
    print(f"Max prediction time: {max_pred_time * 1000:.2f} ms")
    status = 'PASSED' if within_limit else 'FAILED'
    print(f"Time limit ({max_time}s): {status}")


def print_model_info(model_path: str, metadata: Dict[str, Any]) -> None:
    """Print loaded model metadata."""
    print(f"Model: {model_path}")
    print(f"Training subject: {metadata.get('subject', 'unknown')}")
    print(f"Training runs:    {metadata.get('runs', 'unknown')}")
    cv_mean = metadata.get('cv_mean')
    if cv_mean is not None:
        print(f"CV accuracy:      {cv_mean:.4f}")
