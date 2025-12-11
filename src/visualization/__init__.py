"""
Visualization package for EEG classification results.

This package provides plotting functions for:
- Cross-validation scores
- Confusion matrices
- Pipeline comparisons
- Learning curves

Modules:
    cv_plots: Cross-validation plotting (plot_cv_scores, plot_cv_detailed)
    metrics_plots: Metrics visualization (plot_confusion_matrix, plot_training_summary)
    comparison_plots: Pipeline comparison plots (plot_pipeline_comparison)
"""

# Import base utilities
from visualization._base import _finalize_plot

# Import and re-export all plotting functions
from visualization.cv_plots import plot_cv_scores, plot_cv_detailed
from visualization.metrics_plots import plot_confusion_matrix, plot_training_summary
from visualization.comparison_plots import plot_pipeline_comparison

__all__ = [
    '_finalize_plot',
    'plot_cv_scores',
    'plot_cv_detailed',
    'plot_confusion_matrix',
    'plot_training_summary',
    'plot_pipeline_comparison',
]
