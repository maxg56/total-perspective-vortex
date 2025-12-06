"""
Visualization module for EEG classification results.

Provides plotting functions for:
- Cross-validation scores
- Confusion matrices
- Pipeline comparisons
- Learning curves
"""

import numpy as np
import os
import matplotlib

# Use non-interactive backend only in headless environments
# This allows interactive plotting when DISPLAY is available
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List, Optional

from constants import TARGET_ACCURACY


def _finalize_plot(fig, save_path: Optional[str] = None, show: bool = False):
    """Helper function to handle plot saving and closing."""
    if save_path:
        # Create directory if needed
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [Plot saved: {save_path}]")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


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
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot for each fold
    folds = np.arange(1, len(scores) + 1)
    bars = ax.bar(folds, scores, alpha=0.7, color='steelblue', edgecolor='black')

    # Add mean line
    mean_score = scores.mean()
    ax.axhline(mean_score, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_score:.4f}')

    # Add target line
    ax.axhline(TARGET_ACCURACY, color='green', linestyle='--', linewidth=2,
               label=f'Target: {TARGET_ACCURACY:.2f}')

    # Styling
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xticks(folds)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (fold, score) in enumerate(zip(folds, scores)):
        ax.text(fold, score + 0.02, f'{score:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)


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
            text = ax.text(j, i, str(cm[i, j]),
                          ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white",
                          fontsize=14, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_pipeline_comparison(results: Dict[str, Dict],
                             title: str = "Pipeline Comparison",
                             save_path: Optional[str] = None,
                             show: bool = False):
    """
    Plot comparison of multiple pipelines.

    Parameters
    ----------
    results : dict
        Dictionary mapping pipeline names to their scores
        Format: {name: {'mean': float, 'std': float, 'scores': array}}
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    # Filter out failed pipelines
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("No valid results to plot")
        return None

    # Sort by mean score
    sorted_results = sorted(valid_results.items(),
                           key=lambda x: x[1]['mean'],
                           reverse=True)

    names = [item[0] for item in sorted_results]
    means = [item[1]['mean'] for item in sorted_results]
    stds = [item[1]['std'] for item in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar plot with error bars
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, means, yerr=stds, alpha=0.7,
                  color='steelblue', edgecolor='black',
                  error_kw={'linewidth': 2, 'ecolor': 'red'})

    # Add target line
    ax.axhline(TARGET_ACCURACY, color='green', linestyle='--', linewidth=2,
               label=f'Target: {TARGET_ACCURACY:.2f}')

    # Styling
    ax.set_xlabel('Pipeline', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Highlight best pipeline
    best_idx = 0  # Already sorted
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2)

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
    """
    # Filter out failed pipelines
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("No valid results to plot")
        return None

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create box plots
    data = [v['scores'] for v in valid_results.values()]
    names = list(valid_results.keys())

    bp = ax.boxplot(data, labels=names, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Add target line
    ax.axhline(TARGET_ACCURACY, color='green', linestyle='--', linewidth=2,
               label=f'Target: {TARGET_ACCURACY:.2f}')

    # Styling
    ax.set_xlabel('Pipeline', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)


def plot_training_summary(cv_scores: np.ndarray,
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
    """
    fig = plt.figure(figsize=(15, 5))

    # Subplot 1: CV Scores
    ax1 = plt.subplot(1, 3, 1)
    folds = np.arange(1, len(cv_scores) + 1)
    bars = ax1.bar(folds, cv_scores, alpha=0.7, color='steelblue',
                   edgecolor='black')
    mean_score = cv_scores.mean()
    ax1.axhline(mean_score, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_score:.4f}')
    ax1.axhline(0.60, color='green', linestyle='--', linewidth=2,
               label='Target: 0.60')
    ax1.set_xlabel('Fold', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.set_title('Cross-Validation Scores', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.set_xticks(folds)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    for fold, score in zip(folds, cv_scores):
        ax1.text(fold, score + 0.02, f'{score:.3f}',
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
            ax2.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
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

    bars = ax3.bar(classes, class_acc, alpha=0.7, color='coral',
                   edgecolor='black')
    ax3.axhline(TARGET_ACCURACY, color='green', linestyle='--', linewidth=2,
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
        ax3.text(cls, acc + 0.02, f'{acc:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'Training Summary: {pipeline_name}',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)
