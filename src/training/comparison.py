"""
Pipeline comparison utilities for EEG classification.

Provides functions to compare multiple pipelines.
"""

import logging
from typing import Dict, Optional, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score, StratifiedKFold

import display
from constants import RANDOM_STATE
from pipeline import get_pipeline, list_pipelines
from visualization import plot_pipeline_comparison, plot_cv_detailed

# Configure logging
logger = logging.getLogger(__name__)


def compare_pipelines(
        X: NDArray[np.float64], y: NDArray[np.int64],
        cv: int = 5, verbose: bool = True,
        plot: bool = True, save_plots: bool = False,
        show_plots: bool = False,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Compare all available pipelines.

    Parameters
    ----------
    X : np.ndarray
        EEG data
    y : np.ndarray
        Labels
    cv : int
        Number of cross-validation folds
    verbose : bool
        Whether to print results
    plot : bool
        Whether to plot results
    save_plots : bool
        Whether to save plots to disk

    Returns
    -------
    results : dict
        Dictionary mapping pipeline names to their scores
    """
    results: Dict[str, Optional[Dict[str, Any]]] = {}

    if verbose:
        display.section("Comparing all pipelines")

    _, class_counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(cv, int(class_counts.min())))

    for name in list_pipelines():
        try:
            pipeline = get_pipeline(name)
            cv_splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
            )
            scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='accuracy')
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            if verbose:
                display.print_pipeline_result(name, scores)

        except Exception as e:
            logger.warning(f"Pipeline {name} failed: {e}")
            if verbose:
                print(f"\n{name}: FAILED - {e}")
            results[name] = None

    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        raise RuntimeError("All pipelines failed during comparison")
    best = max(valid_results.items(), key=lambda x: x[1]['mean'])
    if verbose:
        display.print_best_pipeline(best[0], best[1]['mean'])

    # Plot results
    if plot:
        # Bar plot comparison
        comp_save = "plots/pipeline_comparison.png" if save_plots else None
        plot_pipeline_comparison(valid_results, save_path=comp_save, show=show_plots)

        # Detailed box plot
        detail_save = "plots/pipeline_comparison_detailed.png" if save_plots else None
        plot_cv_detailed(valid_results, save_path=detail_save, show=show_plots)

    return results
