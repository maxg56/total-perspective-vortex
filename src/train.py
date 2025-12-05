"""
Training module for EEG classification.

Handles:
- Model training with cross-validation
- Hyperparameter selection
- Model saving
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import preprocess_subject, get_run_type
from pipeline import get_pipeline, list_pipelines


def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                       pipeline_name: str = 'csp_lda',
                       cv: int = 5,
                       verbose: bool = True,
                       **pipeline_kwargs) -> tuple:
    """
    Train a pipeline with cross-validation.

    Parameters
    ----------
    X : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times)
    y : np.ndarray
        Labels of shape (n_epochs,)
    pipeline_name : str
        Name of the pipeline to use
    cv : int
        Number of cross-validation folds
    verbose : bool
        Whether to print results
    **pipeline_kwargs : dict
        Additional arguments for the pipeline

    Returns
    -------
    pipeline : Pipeline
        Fitted pipeline (on all data)
    scores : np.ndarray
        Cross-validation scores
    """
    # Build pipeline
    pipeline = get_pipeline(pipeline_name, **pipeline_kwargs)

    # Cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='accuracy')

    if verbose:
        print(f"\n{'='*50}")
        print(f"Pipeline: {pipeline_name}")
        print(f"{'='*50}")
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        print(f"Min: {scores.min():.4f}, Max: {scores.max():.4f}")

        # Check if target achieved
        if scores.mean() >= 0.60:
            print(f"Target accuracy (60%) ACHIEVED")
        else:
            print(f"Target accuracy (60%) NOT achieved - need {0.60 - scores.mean():.4f} more")

    # Fit on all data for final model
    pipeline.fit(X, y)

    return pipeline, scores


def train_with_holdout(X: np.ndarray, y: np.ndarray,
                       pipeline_name: str = 'csp_lda',
                       test_size: float = 0.2,
                       cv: int = 5,
                       verbose: bool = True,
                       **pipeline_kwargs) -> tuple:
    """
    Train a pipeline with holdout test set.

    Parameters
    ----------
    X : np.ndarray
        EEG data
    y : np.ndarray
        Labels
    pipeline_name : str
        Name of the pipeline
    test_size : float
        Proportion of data for test set
    cv : int
        Number of cross-validation folds on training set
    verbose : bool
        Whether to print results
    **pipeline_kwargs : dict
        Additional arguments for the pipeline

    Returns
    -------
    pipeline : Pipeline
        Fitted pipeline
    cv_scores : np.ndarray
        Cross-validation scores on training set
    test_accuracy : float
        Accuracy on holdout test set
    """
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if verbose:
        print(f"\nData split:")
        print(f"  Training: {len(y_train)} epochs")
        print(f"  Test: {len(y_test)} epochs")

    # Build pipeline
    pipeline = get_pipeline(pipeline_name, **pipeline_kwargs)

    # Cross-validation on training set
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring='accuracy')

    if verbose:
        print(f"\nCross-validation on training set:")
        print(f"  Scores: {cv_scores}")
        print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Fit on all training data
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        print(f"\nTest set evaluation:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"\nClassification report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Check if target achieved
        if test_accuracy >= 0.60:
            print(f"\nTarget accuracy (60%) ACHIEVED on test set")
        else:
            print(f"\nTarget accuracy (60%) NOT achieved on test set")

    return pipeline, cv_scores, test_accuracy


def compare_pipelines(X: np.ndarray, y: np.ndarray,
                      cv: int = 5, verbose: bool = True) -> dict:
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

    Returns
    -------
    results : dict
        Dictionary mapping pipeline names to their scores
    """
    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("Comparing all pipelines")
        print("=" * 60)

    for name in list_pipelines():
        try:
            pipeline = get_pipeline(name)
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='accuracy')
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }

            if verbose:
                print(f"\n{name}:")
                print(f"  Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        except Exception as e:
            if verbose:
                print(f"\n{name}: FAILED - {e}")
            results[name] = None

    # Find best pipeline
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1]['mean'])
        if verbose:
            print(f"\nBest pipeline: {best[0]} with {best[1]['mean']:.4f} accuracy")

    return results


def save_model(pipeline, path: str, metadata: dict = None):
    """
    Save trained model to disk.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline
    path : str
        Path to save the model
    metadata : dict
        Additional metadata to save with the model
    """
    model_data = {
        'pipeline': pipeline,
        'metadata': metadata or {}
    }

    # Create directory if needed
    dir_path = os.path.dirname(path)
    if dir_path:  # Only create directory if path includes one
        os.makedirs(dir_path, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved to: {path}")


def load_model(path: str) -> tuple:
    """
    Load trained model from disk.

    Parameters
    ----------
    path : str
        Path to the saved model

    Returns
    -------
    pipeline : Pipeline
        Trained sklearn pipeline
    metadata : dict
        Metadata saved with the model
    """
    with open(path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data['pipeline'], model_data['metadata']


def train_subject(subject: int, runs: list,
                  pipeline_name: str = 'csp_lda',
                  model_dir: str = 'models',
                  cv: int = 5,
                  **pipeline_kwargs) -> tuple:
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
    **pipeline_kwargs : dict
        Additional arguments for the pipeline

    Returns
    -------
    pipeline : Pipeline
        Trained pipeline
    scores : np.ndarray
        Cross-validation scores
    """
    print(f"\n{'='*60}")
    print(f"Training BCI model")
    print(f"Subject: {subject}, Runs: {runs}")
    print(f"Pipeline: {pipeline_name}")
    print(f"{'='*60}")

    # Preprocess data
    print("\nLoading and preprocessing data...")
    X, y, epochs = preprocess_subject(subject, runs)

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # Train and evaluate
    pipeline, scores = train_and_evaluate(
        X, y, pipeline_name=pipeline_name, cv=cv, **pipeline_kwargs
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


if __name__ == "__main__":
    import sys

    # Default: train on subject 1, run 6 (motor imagery: hands vs feet)
    subject = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    print("EEG BCI Training")
    print("=" * 60)

    # Train
    pipeline, scores = train_subject(subject, [run], pipeline_name='csp_lda')

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final CV accuracy: {scores.mean():.4f}")
