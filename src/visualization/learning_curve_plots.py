"""
Learning curve visualization functions.

Provides plots for diagnosing overfitting/underfitting via learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from typing import Optional

from constants import (
    RANDOM_STATE,
    PLOT_FIGSIZE_LEARNING,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_TITLE,
    PLOT_FONTSIZE_LEGEND,
    TARGET_ACCURACY,
)
from visualization._base import _finalize_plot


def plot_learning_curve(pipeline: Pipeline,
                        X: np.ndarray,
                        y: np.ndarray,
                        cv: int = 5,
                        title: str = "Learning Curve",
                        save_path: Optional[str] = None,
                        show: bool = False):
    """
    Plot training and cross-validation accuracy as a function of training size.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The (unfitted) pipeline to evaluate
    X : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times)
    y : np.ndarray
        Labels of shape (n_epochs,)
    cv : int
        Number of cross-validation folds
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to show the figure interactively

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Compute the minimum safe training fraction: stratified CV requires at least
    # 1 sample per class in each fold's training split, so the smallest slice must
    # contain at least (n_classes * cv) samples.
    n_classes = len(np.unique(y))
    n_samples = len(y)
    min_safe = max(0.1, (n_classes * cv) / n_samples)
    train_sizes = np.linspace(min_safe, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=cv_splitter,
        train_sizes=train_sizes,
        scoring='accuracy',
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_LEARNING)

    ax.plot(train_sizes_abs, train_mean, 'o-', color='steelblue',
            label='Training score', linewidth=2)
    ax.fill_between(train_sizes_abs,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='steelblue')

    ax.plot(train_sizes_abs, val_mean, 'o-', color='coral',
            label='Cross-validation score', linewidth=2)
    ax.fill_between(train_sizes_abs,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='coral')

    ax.axhline(TARGET_ACCURACY, color='green', linestyle='--', linewidth=1.5,
               label=f'Target: {TARGET_ACCURACY:.2f}')

    ax.set_xlabel('Training set size', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel('Accuracy', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=PLOT_FONTSIZE_LEGEND)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)
