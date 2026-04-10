"""
Transforms package for EEG signal processing.

This package provides spatial filtering, dimensionality reduction,
and custom linear algebra techniques for EEG-based Brain-Computer Interfaces.

Classes:
    MyCSP: Common Spatial Patterns implementation
    MyPCA: Principal Component Analysis implementation

Modules:
    linalg: Custom linear algebra (eigenvalue, covariance, SVD, Cholesky)
"""

from transforms.csp import MyCSP
from transforms.pca import MyPCA

__all__ = ['MyCSP', 'MyPCA']
