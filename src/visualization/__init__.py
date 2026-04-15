"""
Visualization package for EEG classification results.

This package provides plotting functions for:
- Cross-validation scores
- Confusion matrices
- Pipeline comparisons
- Learning curves
- CSP spatial filters
- Per-class metrics (precision/recall/F1)
- ROC/AUC curves

Modules:
    cv_plots: Cross-validation plotting (plot_cv_scores, plot_cv_detailed)
    metrics_plots: Metrics visualization (plot_confusion_matrix, plot_training_summary)
    comparison_plots: Pipeline comparison plots (plot_pipeline_comparison)
    csp_plots: CSP filter visualization (plot_csp_filters)
    learning_curve_plots: Learning curve plots (plot_learning_curve)
    advanced_metrics_plots: Per-class metrics and ROC (plot_class_metrics, plot_roc_curve)
"""

# Import base utilities
from visualization._base import _finalize_plot

# Import and re-export all plotting functions
from visualization.cv_plots import plot_cv_scores, plot_cv_detailed
from visualization.metrics_plots import plot_confusion_matrix, plot_training_summary
from visualization.comparison_plots import plot_pipeline_comparison
from visualization.csp_plots import plot_csp_filters
from visualization.learning_curve_plots import plot_learning_curve
from visualization.advanced_metrics_plots import plot_class_metrics, plot_roc_curve

__all__ = [
    '_finalize_plot',
    'plot_cv_scores',
    'plot_cv_detailed',
    'plot_confusion_matrix',
    'plot_training_summary',
    'plot_pipeline_comparison',
    'plot_csp_filters',
    'plot_learning_curve',
    'plot_class_metrics',
    'plot_roc_curve',
]
