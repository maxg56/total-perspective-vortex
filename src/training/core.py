"""
Core training functions for EEG classification.

Provides basic training with cross-validation and holdout evaluation.
"""

import logging
from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from constants import RANDOM_STATE, TARGET_ACCURACY, SEPARATOR_WIDTH_NORMAL
from pipeline import get_pipeline
from visualization import (plot_cv_scores, plot_confusion_matrix,
                           plot_training_summary)

# Configure logging
logger = logging.getLogger(__name__)


def train_and_evaluate(X: NDArray[np.float64], y: NDArray[np.int64],
                       pipeline_name: str = 'csp_lda',
                       cv: int = 5,
                       verbose: bool = True,
                       plot: bool = True,
                       save_plots: bool = False,
                       **pipeline_kwargs: Any) -> Tuple[Pipeline, NDArray[np.float64]]:
    """
    Train a pipeline with cross-validation.

    Parameters
    ----------
    X : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times)
    y : np.ndarray
        Labels of shape (n_epochs,)
    pipeline_name : str
        Name of the pipeline to use
    cv : int
        Number of cross-validation folds
    verbose : bool
        Whether to print results
    plot : bool
        Whether to plot results
    save_plots : bool
        Whether to save plots to disk
    **pipeline_kwargs : dict
        Additional arguments for the pipeline

    Returns
    -------
    pipeline : Pipeline
        Fitted pipeline (on all data)
    scores : np.ndarray
        Cross-validation scores
    """
    # Build pipeline
    pipeline = get_pipeline(pipeline_name, **pipeline_kwargs)

    # Cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='accuracy')

    if verbose:
        sep = '=' * SEPARATOR_WIDTH_NORMAL
        print(f"\n{sep}")
        print(f"Pipeline: {pipeline_name}")
        print(f"{sep}")
        print(f"Cross-validation scores: {scores}")
        std_dev = scores.std() * 2
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {std_dev:.4f})")
        print(f"Min: {scores.min():.4f}, Max: {scores.max():.4f}")

        # Check if target achieved
        if scores.mean() >= TARGET_ACCURACY:
            print(f"Target accuracy ({TARGET_ACCURACY:.0%}) ACHIEVED")
        else:
            gap = TARGET_ACCURACY - scores.mean()
            print(f"Target accuracy ({TARGET_ACCURACY:.0%}) NOT achieved "
                  f"- need {gap:.4f} more")

    # Fit on all data for final model
    pipeline.fit(X, y)

    # Plot results
    if plot:
        save_path = f"plots/cv_scores_{pipeline_name}.png" if save_plots else None
        plot_cv_scores(scores, title=f"CV Scores: {pipeline_name}", save_path=save_path)

    return pipeline, scores


def train_with_holdout(X: NDArray[np.float64], y: NDArray[np.int64],
                       pipeline_name: str = 'csp_lda',
                       test_size: float = 0.2,
                       cv: int = 5,
                       verbose: bool = True,
                       plot: bool = True,
                       save_plots: bool = False,
                       **pipeline_kwargs: Any) -> Tuple[Pipeline, NDArray[np.float64], float]:
    """
    Train a pipeline with holdout test set.

    Parameters
    ----------
    X : np.ndarray
        EEG data
    y : np.ndarray
        Labels
    pipeline_name : str
        Name of the pipeline
    test_size : float
        Proportion of data for test set
    cv : int
        Number of cross-validation folds on training set
    verbose : bool
        Whether to print results
    plot : bool
        Whether to plot results
    save_plots : bool
        Whether to save plots to disk
    **pipeline_kwargs : dict
        Additional arguments for the pipeline

    Returns
    -------
    pipeline : Pipeline
        Fitted pipeline
    cv_scores : np.ndarray
        Cross-validation scores on training set
    test_accuracy : float
        Accuracy on holdout test set
    """
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    if verbose:
        print("\nData split:")
        print(f"  Training: {len(y_train)} epochs")
        print(f"  Test: {len(y_test)} epochs")

    # Build pipeline
    pipeline = get_pipeline(pipeline_name, **pipeline_kwargs)

    # Cross-validation on training set
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring='accuracy')

    if verbose:
        print("\nCross-validation on training set:")
        print(f"  Scores: {cv_scores}")
        std_dev = cv_scores.std() * 2
        print(f"  Mean: {cv_scores.mean():.4f} (+/- {std_dev:.4f})")

    # Fit on all training data
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        print("\nTest set evaluation:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Check if target achieved
        if test_accuracy >= TARGET_ACCURACY:
            msg = f"\nTarget accuracy ({TARGET_ACCURACY:.0%}) ACHIEVED on test set"
            print(msg)
        else:
            msg = f"\nTarget accuracy ({TARGET_ACCURACY:.0%}) NOT achieved on test set"
            print(msg)

    # Plot results
    if plot:
        # Plot CV scores
        cv_save = f"plots/cv_scores_{pipeline_name}_holdout.png" if save_plots else None
        cv_title = f"CV Scores (Training Set): {pipeline_name}"
        plot_cv_scores(cv_scores, title=cv_title, save_path=cv_save)

        # Plot confusion matrix
        cm_save = f"plots/confusion_matrix_{pipeline_name}.png" if save_plots else None
        cm_title = f"Confusion Matrix (Test Set): {pipeline_name}"
        plot_confusion_matrix(
            y_test, y_pred,
            title=cm_title,
            save_path=cm_save)

        # Plot comprehensive summary
        summary_save = f"plots/training_summary_{pipeline_name}.png" if save_plots else None
        plot_training_summary(
            cv_scores, y_test, y_pred,
            pipeline_name=pipeline_name,
            save_path=summary_save)

    return pipeline, cv_scores, test_accuracy
