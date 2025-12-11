"""
Base utilities for visualization.

Internal module providing shared functionality for plotting.
"""

import os
import matplotlib

# Use non-interactive backend only in headless environments
# This allows interactive plotting when DISPLAY is available
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from typing import Optional

from constants import PLOT_DPI


def _finalize_plot(fig, save_path: Optional[str] = None, show: bool = False):
    """
    Handle plot saving and closing.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to finalize
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to show the figure interactively

    Returns
    -------
    fig : matplotlib.figure.Figure
        The finalized figure
    """
    if save_path:
        # Create directory if needed
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"  [Plot saved: {save_path}]")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
