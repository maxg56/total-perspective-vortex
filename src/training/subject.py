"""
Subject-specific training pipeline for EEG classification.

Provides complete training workflow for individual subjects.
"""

import os
from typing import Tuple, List, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from preprocess import preprocess_subject, get_run_type
from training.core import train_and_evaluate
from training.persistence import save_model


def train_subject(subject: int, runs: List[int],
                  pipeline_name: str = 'csp_lda',
                  model_dir: str = 'models',
                  cv: int = 5,
                  plot: bool = True,
                  save_plots: bool = False,
                  **pipeline_kwargs: Any) -> Tuple[Pipeline, NDArray[np.float64]]:
    """
    Complete training pipeline for a subject.

    Parameters
    ----------
    subject : int
        Subject number
    runs : list
        List of run numbers
    pipeline_name : str
        Name of the pipeline to use
    model_dir : str
        Directory to save the model
    cv : int
        Number of cross-validation folds
    plot : bool
        Whether to plot results
    save_plots : bool
        Whether to save plots to disk
    **pipeline_kwargs : dict
        Additional arguments for the pipeline

    Returns
    -------
    pipeline : Pipeline
        Trained pipeline
    scores : np.ndarray
        Cross-validation scores
    """
    sep = '=' * 60
    print(f"\n{sep}")
    print("Training BCI model")
    print(f"Subject: {subject}, Runs: {runs}")
    print(f"Pipeline: {pipeline_name}")
    print(f"{sep}")

    # Preprocess data
    print("\nLoading and preprocessing data...")
    X, y, epochs = preprocess_subject(subject, runs)

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Class distribution: {class_dist}")

    # Train and evaluate
    pipeline, scores = train_and_evaluate(
        X, y, pipeline_name=pipeline_name, cv=cv,
        plot=plot, save_plots=save_plots, **pipeline_kwargs
    )

    # Save model
    run_type = get_run_type(runs[0])
    model_path = os.path.join(model_dir, f"model_s{subject}_r{runs[0]}_{pipeline_name}.pkl")

    metadata = {
        'subject': subject,
        'runs': runs,
        'run_type': run_type,
        'pipeline_name': pipeline_name,
        'cv_scores': scores.tolist(),
        'cv_mean': scores.mean(),
        'n_epochs': len(y),
        'n_channels': X.shape[1],
        'n_times': X.shape[2],
        'classes': np.unique(y).tolist()
    }

    save_model(pipeline, model_path, metadata)

    return pipeline, scores
