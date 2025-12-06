#!/usr/bin/env python
"""
Demo script showing visualization capabilities.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess_subject
from train import train_with_holdout


def main():
    # Get subject and run from command line or use defaults
    subject = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run = int(sys.argv[2]) if len(sys.argv) > 2 else 14

    print("=" * 70)
    print("EEG BCI Visualization Demo")
    print("=" * 70)
    print(f"Subject: {subject}")
    print(f"Run: {run}")
    print()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, epochs = preprocess_subject(subject, [run])
    print(f"Data shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print()

    # Train with holdout and generate all visualizations
    print("Training with holdout validation...")
    print("Generating visualizations...")
    print()

    pipeline, cv_scores, test_accuracy = train_with_holdout(
        X, y,
        pipeline_name='csp_lda',
        test_size=0.2,
        cv=5,
        verbose=True,
        plot=True,
        save_plots=True
    )

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Cross-validation mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Test set accuracy: {test_accuracy:.4f}")
    print()
    print("Plots saved to:")
    print("  - plots/cv_scores_csp_lda_holdout.png")
    print("  - plots/confusion_matrix_csp_lda.png")
    print("  - plots/training_summary_csp_lda.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
