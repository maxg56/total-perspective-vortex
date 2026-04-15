"""
CSP spatial filter visualization functions.

Provides plots for inspecting the learned CSP filters and eigenvalues.
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from constants import (
    PLOT_FIGSIZE_CSP,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_TITLE,
    PLOT_FONTSIZE_LEGEND,
    PLOT_FONTSIZE_SMALL,
    PLOT_TEXT_OFFSET,
)
from visualization._base import _finalize_plot


def plot_csp_filters(W: np.ndarray,
                     eigenvalues: np.ndarray,
                     channel_names: Optional[List[str]] = None,
                     title: str = "CSP Spatial Filters",
                     save_path: Optional[str] = None,
                     show: bool = False):
    """
    Plot CSP spatial filter weights and their associated eigenvalues.

    Parameters
    ----------
    W : np.ndarray
        CSP filter matrix of shape (n_channels, n_components)
    eigenvalues : np.ndarray
        Eigenvalues of shape (n_components,)
    channel_names : list of str, optional
        EEG channel names for y-axis labels
    title : str
        Overall figure title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to show the figure interactively

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    n_channels, n_components = W.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_FIGSIZE_CSP)

    # --- Left panel: filter weight heatmap ---
    im = ax1.imshow(W, cmap='RdBu_r', aspect='auto',
                    vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    plt.colorbar(im, ax=ax1, label='Filter weight')

    ax1.set_xlabel('CSP Component', fontsize=PLOT_FONTSIZE_LABEL)
    ax1.set_ylabel('Channel', fontsize=PLOT_FONTSIZE_LABEL)
    ax1.set_title('Filter Weights', fontsize=PLOT_FONTSIZE_LABEL, fontweight='bold')
    ax1.set_xticks(np.arange(n_components))
    ax1.set_xticklabels([f'C{i + 1}' for i in range(n_components)],
                        fontsize=PLOT_FONTSIZE_SMALL)
    if channel_names is not None:
        ax1.set_yticks(np.arange(n_channels))
        ax1.set_yticklabels(channel_names, fontsize=PLOT_FONTSIZE_SMALL)
    else:
        ax1.set_yticks(np.arange(0, n_channels, max(1, n_channels // 10)))

    # --- Right panel: eigenvalue bar chart ---
    comp_idx = np.arange(1, n_components + 1)
    # Color: high eigenvalues (class-1 discriminative) in blue, low in red
    colors = matplotlib.colormaps['RdYlBu'](np.linspace(0.1, 0.9, n_components))[::-1]
    bars = ax2.bar(comp_idx, eigenvalues, color=colors, edgecolor='black', alpha=0.85)

    # Label each bar
    for bar, val in zip(bars, eigenvalues):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + PLOT_TEXT_OFFSET,
                 f'{val:.3f}',
                 ha='center', va='bottom', fontsize=PLOT_FONTSIZE_SMALL)

    ax2.axhline(0.5, color='grey', linestyle='--', linewidth=1,
                label='Chance (0.5)', alpha=0.7)
    ax2.set_xlabel('CSP Component', fontsize=PLOT_FONTSIZE_LABEL)
    ax2.set_ylabel('Eigenvalue', fontsize=PLOT_FONTSIZE_LABEL)
    ax2.set_title('Eigenvalues', fontsize=PLOT_FONTSIZE_LABEL, fontweight='bold')
    ax2.set_xticks(comp_idx)
    ax2.set_xticklabels([f'C{i}' for i in comp_idx], fontsize=PLOT_FONTSIZE_SMALL)
    ax2.legend(fontsize=PLOT_FONTSIZE_LEGEND)
    ax2.set_ylim(0, max(1.0, eigenvalues.max() * 1.15))
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    return _finalize_plot(fig, save_path, show)
