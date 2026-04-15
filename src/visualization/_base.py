"""
Base utilities for visualization.

Internal module providing shared functionality for plotting.
"""

import os
import sys
import matplotlib

# Select backend before importing pyplot (cannot be changed after first import).
# Use an interactive backend only when --show-plots is explicitly requested AND
# a display is available; otherwise stay with the headless Agg backend.
_WANT_SHOW = '--show-plots' in sys.argv
_HAS_DISPLAY = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

if _WANT_SHOW and _HAS_DISPLAY:
    # Verify each backend by actually importing its module before selecting it,
    # because matplotlib.use() succeeds silently even when the backend is broken.
    import importlib as _importlib
    _INTERACTIVE_BACKENDS = [
        ('TkAgg', 'matplotlib.backends.backend_tkagg'),
        ('QtAgg', 'matplotlib.backends.backend_qtagg'),
        ('Qt5Agg', 'matplotlib.backends.backend_qt5agg'),
        ('GTK4Agg', 'matplotlib.backends.backend_gtk4agg'),
        ('GTK3Agg', 'matplotlib.backends.backend_gtk3agg'),
        ('WXAgg', 'matplotlib.backends.backend_wxagg'),
    ]
    for _backend, _module in _INTERACTIVE_BACKENDS:
        try:
            _importlib.import_module(_module)
            matplotlib.use(_backend)
            break
        except Exception:
            continue
    else:
        matplotlib.use('Agg')
        _WANT_SHOW = False
        print(
            "Warning: --show-plots requested but no interactive backend is available "
            "(Tk, Qt, GTK, WX). Plots will be saved only.\n"
            "Install a GUI toolkit to enable interactive display, e.g.:\n"
            "  sudo pacman -S python-pyqt6   # Arch Linux\n"
            "  sudo apt install python3-pyqt5 # Debian/Ubuntu"
        )
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
from typing import Optional  # noqa: E402

from constants import PLOT_DPI  # noqa: E402


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

    if show and _WANT_SHOW:
        plt.show()
    else:
        plt.close(fig)

    return fig
