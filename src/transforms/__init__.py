"""
Transforms package for EEG signal processing.

This package provides spatial filtering and dimensionality reduction
techniques for EEG-based Brain-Computer Interfaces.

Classes:
    MyCSP: Common Spatial Patterns implementation
    MyPCA: Principal Component Analysis implementation
"""

from transforms.csp import MyCSP
from transforms.pca import MyPCA

__all__ = ['MyCSP', 'MyPCA']
