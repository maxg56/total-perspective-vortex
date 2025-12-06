"""
Global constants for the Total Perspective Vortex project.

This module centralizes magic numbers and configuration values
used across multiple modules to ensure consistency and maintainability.
"""

from typing import Final

# Numerical stability
EPSILON: Final[float] = 1e-10
EPSILON_SMALL: Final[float] = 1e-6

# Random seed for reproducibility
RANDOM_STATE: Final[int] = 42

# Subject constraints (Physionet EEGMMIDB dataset)
MIN_SUBJECT: Final[int] = 1
MAX_SUBJECT: Final[int] = 109

# Valid run numbers for the dataset
VALID_RUNS: Final[tuple] = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

# Performance target
TARGET_ACCURACY: Final[float] = 0.60

# Prediction time limit (seconds)
MAX_PREDICTION_TIME: Final[float] = 2.0
