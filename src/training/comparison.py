"""
Pipeline comparison utilities for EEG classification.

Provides functions to compare multiple pipelines.
"""

import logging
from typing import Dict, Optional, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score, StratifiedKFold

from constants import RANDOM_STATE
from pipeline import get_pipeline, list_pipelines
from visualization import plot_pipeline_comparison, plot_cv_detailed

# Configure logging
logger = logging.getLogger(__name__)


def compare_pipelines(
        X: NDArray[np.float64], y: NDArray[np.int64],
        cv: int = 5, verbose: bool = True,
        plot: bool = True, save_plots: bool = False
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
        print("\n" + "=" * 60)
        print("Comparing all pipelines")
        print("=" * 60)

    for name in list_pipelines():
        try:
            pipeline = get_pipeline(name)
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='accuracy')
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }

            if verbose:
                print(f"\n{name}:")
                print(f"  Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        except (ValueError, RuntimeError, TypeError) as e:
            logger.warning(f"Pipeline {name} failed: {e}")
            if verbose:
                print(f"\n{name}: FAILED - {e}")
            results[name] = None

    # Find best pipeline
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1]['mean'])
        if verbose:
            best_acc = best[1]['mean']
            print(f"\nBest pipeline: {best[0]} with {best_acc:.4f} accuracy")

    # Plot results
    if plot and valid_results:
        # Bar plot comparison
        comp_save = "plots/pipeline_comparison.png" if save_plots else None
        plot_pipeline_comparison(results, save_path=comp_save)

        # Detailed box plot
        detail_save = "plots/pipeline_comparison_detailed.png" if save_plots else None
        plot_cv_detailed(results, save_path=detail_save)

    return results
