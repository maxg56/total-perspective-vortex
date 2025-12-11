"""
Metrics visualization functions.

Provides plotting for confusion matrices and training summaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import List, Optional

from constants import TARGET_ACCURACY
from visualization._base import _finalize_plot


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None,
                          show: bool = False):
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Names of classes
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
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use matplotlib imshow for heatmap
    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=12)

    # Set ticks and labels
    classes = class_names or np.unique(y_true)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt_color = "black" if cm[i, j] < cm.max() / 2 else "white"
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", color=txt_color,
                fontsize=14, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_training_summary(
        cv_scores: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        pipeline_name: str = "Pipeline",
        save_path: Optional[str] = None,
        show: bool = False):
    """
    Create a comprehensive summary plot with multiple subplots.

    Parameters
    ----------
    cv_scores : np.ndarray
        Cross-validation scores
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Names of classes
    pipeline_name : str
        Name of the pipeline
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to show the figure interactively

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    fig = plt.figure(figsize=(15, 5))

    # Subplot 1: CV Scores
    ax1 = plt.subplot(1, 3, 1)
    folds = np.arange(1, len(cv_scores) + 1)
    ax1.bar(folds, cv_scores, alpha=0.7, color='steelblue',
            edgecolor='black')
    mean_score = cv_scores.mean()
    ax1.axhline(
        mean_score, color='red', linestyle='--', linewidth=2,
        label=f'Mean: {mean_score:.4f}')
    ax1.axhline(
        0.60, color='green', linestyle='--', linewidth=2,
        label='Target: 0.60')
    ax1.set_xlabel('Fold', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.set_title('Cross-Validation Scores', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.set_xticks(folds)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    for fold, score in zip(folds, cv_scores):
        ax1.text(
            fold, score + 0.02, f'{score:.3f}',
            ha='center', va='bottom', fontsize=8)

    # Subplot 2: Confusion Matrix
    ax2 = plt.subplot(1, 3, 2)
    cm = confusion_matrix(y_true, y_pred)

    # Use matplotlib imshow for heatmap
    im = ax2.imshow(cm, cmap='Blues', aspect='auto')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Count', fontsize=10)

    # Set ticks and labels
    classes = class_names or np.unique(y_true)
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt_color = "black" if cm[i, j] < cm.max() / 2 else "white"
            ax2.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color=txt_color,
                fontsize=10, fontweight='bold')

    ax2.set_xlabel('Predicted Label', fontsize=10)
    ax2.set_ylabel('True Label', fontsize=10)
    ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

    # Subplot 3: Accuracy per class
    ax3 = plt.subplot(1, 3, 3)
    classes = np.unique(y_true)
    class_acc = []
    for cls in classes:
        mask = y_true == cls
        acc = (y_pred[mask] == cls).mean()
        class_acc.append(acc)

    ax3.bar(classes, class_acc, alpha=0.7, color='coral',
            edgecolor='black')
    ax3.axhline(
        TARGET_ACCURACY, color='green', linestyle='--', linewidth=2,
        label=f'Target: {TARGET_ACCURACY:.2f}')
    ax3.set_xlabel('Class', fontsize=10)
    ax3.set_ylabel('Accuracy', fontsize=10)
    ax3.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.set_xticks(classes)
    if class_names:
        ax3.set_xticklabels(class_names)
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    for cls, acc in zip(classes, class_acc):
        ax3.text(
            cls, acc + 0.02, f'{acc:.3f}',
            ha='center', va='bottom', fontsize=8)

    plt.suptitle(
        f'Training Summary: {pipeline_name}',
        fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)
