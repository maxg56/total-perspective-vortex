"""
Core training functions for EEG classification.

Provides basic training with cross-validation and holdout evaluation.
"""

import logging
from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     StratifiedKFold, train_test_split)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import display
from constants import RANDOM_STATE, TARGET_ACCURACY
from pipeline import get_pipeline
from visualization import (plot_cv_scores, plot_confusion_matrix,
                           plot_training_summary, plot_csp_filters,
                           plot_learning_curve, plot_class_metrics,
                           plot_roc_curve)

# Configure logging
logger = logging.getLogger(__name__)


def train_and_evaluate(X: NDArray[np.float64], y: NDArray[np.int64],
                       pipeline_name: str = 'csp_lda',
                       cv: int = 5,
                       verbose: bool = True,
                       plot: bool = True,
                       save_plots: bool = False,
                       show_plots: bool = False,
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
    # Build pipeline (reused for all CV calls; sklearn clones internally)
    pipeline = get_pipeline(pipeline_name, **pipeline_kwargs)

    # Cross-validation — cap folds to min class count to avoid degenerate splits
    _, class_counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(cv, int(class_counts.min())))
    if n_splits < cv:
        logger.warning(
            "Requested %d CV folds but only %d available (min class count = %d). "
            "Using %d folds.", cv, n_splits, int(class_counts.min()), n_splits
        )
    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='accuracy')

    if verbose:
        display.print_cv_result(scores, pipeline_name, TARGET_ACCURACY)

    # Fit on all data for final model
    pipeline.fit(X, y)

    # Plot results
    if plot:
        save_path = f"plots/cv_scores_{pipeline_name}.png" if save_plots else None
        plot_cv_scores(scores, title=f"CV Scores: {pipeline_name}",
                       save_path=save_path, show=show_plots)

        # CSP spatial filters (only when the pipeline contains a CSP step)
        if 'csp' in pipeline.named_steps:
            csp_step = pipeline.named_steps['csp']
            csp_save = f"plots/csp_filters_{pipeline_name}.png" if save_plots else None
            plot_csp_filters(
                csp_step.W_, csp_step.eigenvalues_,
                title=f"CSP Filters: {pipeline_name}",
                save_path=csp_save, show=show_plots,
            )

        # Learning curves — pipeline is cloned internally by learning_curve
        try:
            lc_save = f"plots/learning_curve_{pipeline_name}.png" if save_plots else None
            plot_learning_curve(
                pipeline, X, y, cv=n_splits,
                title=f"Learning Curve: {pipeline_name}",
                save_path=lc_save, show=show_plots,
            )
        except Exception as e:
            logger.warning("Learning curve plot skipped: %s", e)

        # OOF predictions for per-class metrics and ROC — pipeline is cloned by cross_val_predict
        try:
            y_pred_oof = cross_val_predict(pipeline, X, y, cv=cv_splitter)
            metrics_save = f"plots/class_metrics_{pipeline_name}.png" if save_plots else None
            plot_class_metrics(
                y, y_pred_oof,
                title=f"Per-Class Metrics: {pipeline_name}",
                save_path=metrics_save, show=show_plots,
            )

            # ROC curve — try decision_function first, then predict_proba
            roc_save = f"plots/roc_curve_{pipeline_name}.png" if save_plots else None
            y_score = None
            try:
                y_score = cross_val_predict(
                    pipeline, X, y, cv=cv_splitter, method='decision_function'
                )
            except (AttributeError, ValueError):
                try:
                    y_proba = cross_val_predict(
                        pipeline, X, y, cv=cv_splitter, method='predict_proba'
                    )
                    y_score = y_proba[:, 1]
                except (AttributeError, ValueError):
                    pass

            if y_score is not None:
                plot_roc_curve(
                    y, y_score,
                    title=f"ROC Curve: {pipeline_name}",
                    save_path=roc_save, show=show_plots,
                )
        except Exception as e:
            logger.warning("OOF metrics plots skipped: %s", e)

    return pipeline, scores


def train_with_holdout(X: NDArray[np.float64], y: NDArray[np.int64],
                       pipeline_name: str = 'csp_lda',
                       test_size: float = 0.2,
                       cv: int = 5,
                       verbose: bool = True,
                       plot: bool = True,
                       save_plots: bool = False,
                       show_plots: bool = False,
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
        logger.info("Data split: training=%d epochs, test=%d epochs",
                    len(y_train), len(y_test))

    # Build pipeline
    pipeline = get_pipeline(pipeline_name, **pipeline_kwargs)

    # Cross-validation on training set — cap folds to min class count
    _, train_class_counts = np.unique(y_train, return_counts=True)
    n_splits = max(2, min(cv, int(train_class_counts.min())))
    if n_splits < cv:
        logger.warning(
            "Requested %d CV folds but only %d available (min class count = %d). "
            "Using %d folds.", cv, n_splits, int(train_class_counts.min()), n_splits
        )
    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring='accuracy')

    if verbose:
        display.print_cv_result(cv_scores, f"{pipeline_name} (training CV)", TARGET_ACCURACY)

    # Fit on all training data
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        logger.info("Test set accuracy: %.4f", test_accuracy)
        logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
        logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

        if test_accuracy >= TARGET_ACCURACY:
            logger.info("Target accuracy (%s) ACHIEVED on test set", f"{TARGET_ACCURACY:.0%}")
        else:
            logger.info("Target accuracy (%s) NOT achieved on test set",
                        f"{TARGET_ACCURACY:.0%}")

    # Plot results
    if plot:
        # Plot CV scores
        cv_save = f"plots/cv_scores_{pipeline_name}_holdout.png" if save_plots else None
        cv_title = f"CV Scores (Training Set): {pipeline_name}"
        plot_cv_scores(cv_scores, title=cv_title, save_path=cv_save, show=show_plots)

        # Plot confusion matrix
        cm_save = f"plots/confusion_matrix_{pipeline_name}.png" if save_plots else None
        cm_title = f"Confusion Matrix (Test Set): {pipeline_name}"
        plot_confusion_matrix(
            y_test, y_pred,
            title=cm_title,
            save_path=cm_save, show=show_plots)

        # Plot comprehensive summary
        summary_save = f"plots/training_summary_{pipeline_name}.png" if save_plots else None
        plot_training_summary(
            cv_scores, y_test, y_pred,
            pipeline_name=pipeline_name,
            save_path=summary_save, show=show_plots)

    return pipeline, cv_scores, test_accuracy
