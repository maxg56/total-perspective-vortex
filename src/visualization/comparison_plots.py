"""
Pipeline comparison plotting functions.

Provides visualization for comparing multiple pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

from constants import (
    TARGET_ACCURACY,
    PLOT_FIGSIZE_COMPARISON,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_TITLE,
    PLOT_FONTSIZE_LEGEND,
    PLOT_FONTSIZE_ANNOTATION,
    PLOT_TEXT_OFFSET,
    PLOT_XTICK_ROTATION,
)
from visualization._base import _finalize_plot


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

    # Sort by mean score
    sorted_results = sorted(
        valid_results.items(),
        key=lambda x: x[1]['mean'],
        reverse=True)

    names = [item[0] for item in sorted_results]
    means = [item[1]['mean'] for item in sorted_results]
    stds = [item[1]['std'] for item in sorted_results]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_COMPARISON)

    # Bar plot with error bars
    x_pos = np.arange(len(names))
    bars = ax.bar(
        x_pos, means, yerr=stds, alpha=0.7,
        color='steelblue', edgecolor='black',
        error_kw={'linewidth': 2, 'ecolor': 'red'})

    # Add target line
    ax.axhline(TARGET_ACCURACY, color='green', linestyle='--', linewidth=2,
               label=f'Target: {TARGET_ACCURACY:.2f}')

    # Styling
    ax.set_xlabel('Pipeline', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel('Mean Accuracy', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=PLOT_XTICK_ROTATION, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=PLOT_FONTSIZE_LEGEND)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + PLOT_TEXT_OFFSET, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=PLOT_FONTSIZE_ANNOTATION)

    # Highlight best pipeline
    best_idx = 0  # Already sorted
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)
