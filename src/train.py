"""
Training module for EEG classification.

This module provides backwards compatibility. The actual implementations
have been moved to the training package:
    - training.core: train_and_evaluate, train_with_holdout
    - training.comparison: compare_pipelines
    - training.persistence: save_model, load_model
    - training.subject: train_subject

For new code, prefer importing directly from the training package:
    from training import train_and_evaluate, train_subject
"""

# Re-export from training package for backwards compatibility
from training import (
    train_and_evaluate,
    train_with_holdout,
    compare_pipelines,
    save_model,
    load_model,
    train_subject,
)

__all__ = [
    'train_and_evaluate',
    'train_with_holdout',
    'compare_pipelines',
    'save_model',
    'load_model',
    'train_subject',
]


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
