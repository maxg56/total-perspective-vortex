"""
Advanced metrics visualization functions.

Provides per-class precision/recall/F1 plots and ROC/AUC curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from typing import List, Optional

from constants import (
    PLOT_FIGSIZE_METRICS,
    PLOT_FIGSIZE_ROC,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_TITLE,
    PLOT_FONTSIZE_LEGEND,
    PLOT_FONTSIZE_SMALL,
    PLOT_TEXT_OFFSET,
    PLOT_XTICK_ROTATION,
)
from visualization._base import _finalize_plot


def plot_class_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       title: str = "Per-Class Metrics",
                       save_path: Optional[str] = None,
                       show: bool = False):
    """
    Plot precision, recall, and F1-score for each class.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list of str, optional
        Human-readable class names
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
    classes = np.unique(y_true)
    labels = class_names if class_names else [str(c) for c in classes]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes
    )

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_METRICS)

    bars_p = ax.bar(x - width, precision, width, label='Precision',
                    color='steelblue', alpha=0.85, edgecolor='black')
    bars_r = ax.bar(x, recall, width, label='Recall',
                    color='coral', alpha=0.85, edgecolor='black')
    bars_f = ax.bar(x + width, f1, width, label='F1-Score',
                    color='mediumseagreen', alpha=0.85, edgecolor='black')

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + PLOT_TEXT_OFFSET,
                    f'{bar.get_height():.2f}',
                    ha='center', va='bottom', fontsize=PLOT_FONTSIZE_SMALL)

    ax.set_xlabel('Class', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel('Score', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=PLOT_FONTSIZE_LEGEND)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_roc_curve(y_true: np.ndarray,
                   y_score: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   title: str = "ROC Curve",
                   save_path: Optional[str] = None,
                   show: bool = False):
    """
    Plot ROC curve with AUC for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Decision function scores or predicted probabilities for the positive class.
        For decision_function output with shape (n_samples,) or (n_samples, 1).
        For predict_proba output, pass the positive-class column.
    class_names : list of str, optional
        Names for the two classes (used in legend label)
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
    y_score = np.asarray(y_score).ravel()

    pos_label = np.unique(y_true)[-1]
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    label_str = str(class_names[1]) if class_names else str(pos_label)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_ROC)

    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f'{label_str} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--',
            linewidth=1.5, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')

    ax.set_xlabel('False Positive Rate', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel('True Positive Rate', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight='bold')
    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((-0.02, 1.05))
    ax.legend(fontsize=PLOT_FONTSIZE_LEGEND, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)
