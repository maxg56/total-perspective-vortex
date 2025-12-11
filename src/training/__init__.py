"""
Training package for EEG classification.

This package provides training, evaluation, and model persistence
functions for EEG-based Brain-Computer Interfaces.

Modules:
    core: Core training functions (train_and_evaluate, train_with_holdout)
    comparison: Pipeline comparison utilities
    persistence: Model saving and loading
    subject: Subject-specific training pipeline
"""

from training.core import train_and_evaluate, train_with_holdout
from training.comparison import compare_pipelines
from training.persistence import save_model, load_model
from training.subject import train_subject

__all__ = [
    'train_and_evaluate',
    'train_with_holdout',
    'compare_pipelines',
    'save_model',
    'load_model',
    'train_subject',
]
