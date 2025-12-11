"""
Cross-validation plotting functions.

Provides visualization for cross-validation scores.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

from constants import (
    TARGET_ACCURACY,
    PLOT_FIGSIZE_CV,
    PLOT_FIGSIZE_DETAILED,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_TITLE,
    PLOT_FONTSIZE_LEGEND,
    PLOT_XTICK_ROTATION,
)
from visualization._base import _finalize_plot


def plot_cv_scores(scores: np.ndarray,
                   title: str = "Cross-Validation Scores",
                   save_path: Optional[str] = None,
                   show: bool = False):
    """
    Plot cross-validation scores.

    Parameters
    ----------
    scores : np.ndarray
        Cross-validation scores
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
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_CV)

    # Bar plot for each fold
    folds = np.arange(1, len(scores) + 1)
    ax.bar(folds, scores, alpha=0.7, color='steelblue', edgecolor='black')

    # Add mean line
    mean_score = scores.mean()
    ax.axhline(mean_score, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_score:.4f}')

    # Add target line
    ax.axhline(TARGET_ACCURACY, color='green', linestyle='--', linewidth=2,
               label=f'Target: {TARGET_ACCURACY:.2f}')

    # Styling
    ax.set_xlabel('Fold', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel('Accuracy', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xticks(folds)
    ax.legend(fontsize=PLOT_FONTSIZE_LEGEND)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (fold, score) in enumerate(zip(folds, scores)):
        ax.text(fold, score + 0.02, f'{score:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_cv_detailed(results: Dict[str, Dict],
                     title: str = "Detailed Cross-Validation Scores",
                     save_path: Optional[str] = None,
                     show: bool = False):
    """
    Plot detailed cross-validation scores for multiple pipelines.

    Parameters
    ----------
    results : dict
        Dictionary mapping pipeline names to their scores
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to show the figure interactively

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The generated figure, or None if no valid results
    """
    # Filter out failed pipelines
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("No valid results to plot")
        return None

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_DETAILED)

    # Create box plots
    data = [v['scores'] for v in valid_results.values()]
    names = list(valid_results.keys())

    bp = ax.boxplot(data, tick_labels=names, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Add target line
    ax.axhline(TARGET_ACCURACY, color='green', linestyle='--', linewidth=2,
               label=f'Target: {TARGET_ACCURACY:.2f}')

    # Styling
    ax.set_xlabel('Pipeline', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel('Accuracy', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight='bold')
    ax.set_xticklabels(names, rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=PLOT_FONTSIZE_LEGEND)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)
