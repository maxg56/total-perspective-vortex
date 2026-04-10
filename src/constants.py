"""
Global constants for the Total Perspective Vortex project.

This module centralizes magic numbers and configuration values
used across multiple modules to ensure consistency and maintainability.
"""

from typing import Final, Tuple

# =============================================================================
# Numerical stability
# =============================================================================
EPSILON: Final[float] = 1e-10
EPSILON_SMALL: Final[float] = 1e-6

# =============================================================================
# Random seed for reproducibility
# =============================================================================
RANDOM_STATE: Final[int] = 42

# =============================================================================
# Physionet EEGMMIDB dataset constraints
# =============================================================================
MIN_SUBJECT: Final[int] = 1
MAX_SUBJECT: Final[int] = 109

# Valid run numbers for the dataset
VALID_RUNS: Final[tuple] = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

# Default values for demo/testing
DEFAULT_SUBJECT: Final[int] = 4
DEFAULT_RUN: Final[int] = 14

# =============================================================================
# EEG Signal Processing
# =============================================================================
# Sampling rate for Physionet EEGMMIDB dataset (Hz)
EEG_SAMPLING_RATE: Final[float] = 160.0

# Bandpass filter cutoff frequencies (Hz)
EEG_BANDPASS_LOW: Final[float] = 7.0
EEG_BANDPASS_HIGH: Final[float] = 30.0

# Welch PSD method parameters
WELCH_NPERSEG: Final[int] = 256
WELCH_NOVERLAP: Final[int] = 128

# EEG frequency bands (Hz)
MU_BAND_LOW: Final[float] = 8.0
MU_BAND_HIGH: Final[float] = 12.0
BETA_BAND_LOW: Final[float] = 12.0
BETA_BAND_HIGH: Final[float] = 30.0

# Filter edge case boundaries (normalized frequency)
FILTER_EDGE_LOW: Final[float] = 0.01
FILTER_EDGE_HIGH: Final[float] = 0.99

# =============================================================================
# Test data dimensions
# =============================================================================
TEST_N_EPOCHS: Final[int] = 10
TEST_N_EPOCHS_LARGE: Final[int] = 50
TEST_N_CHANNELS: Final[int] = 64
TEST_N_TIMES: Final[int] = 480

# =============================================================================
# Machine Learning defaults
# =============================================================================
DEFAULT_N_COMPONENTS_PCA: Final[int] = 10
DEFAULT_N_COMPONENTS_PCA_PIPELINE: Final[int] = 50

# =============================================================================
# Performance targets
# =============================================================================
TARGET_ACCURACY: Final[float] = 0.60

# Per-experiment accuracy targets (6 experiments across 109 subjects)
# Experiments group runs into consecutive pairs:
#   Exp 0: runs 3,4  (left/right, repetition 1)
#   Exp 1: runs 5,6  (fists/feet, repetition 1)
#   Exp 2: runs 7,8  (left/right, repetition 2)
#   Exp 3: runs 9,10 (fists/feet, repetition 2)
#   Exp 4: runs 11,12 (left/right, repetition 3)
#   Exp 5: runs 13,14 (fists/feet, repetition 3)
EXPERIMENT_TARGETS: Final[dict] = {
    0: 0.59,
    1: 0.57,
    2: 0.71,
    3: 0.60,
    4: 0.59,
    5: 0.67,
}

# Mapping from run number to experiment index
RUN_TO_EXPERIMENT: Final[dict] = {
    3: 0, 4: 0,
    5: 1, 6: 1,
    7: 2, 8: 2,
    9: 3, 10: 3,
    11: 4, 12: 4,
    13: 5, 14: 5,
}

# Prediction time limit (seconds)
MAX_PREDICTION_TIME: Final[float] = 2.0

# =============================================================================
# Visualization constants
# =============================================================================
# DPI for saved plots
PLOT_DPI: Final[int] = 300

# Text separator widths
SEPARATOR_WIDTH: Final[int] = 60
SEPARATOR_WIDTH_WIDE: Final[int] = 70
SEPARATOR_WIDTH_NORMAL: Final[int] = 50

# Font sizes
PLOT_FONTSIZE_LABEL: Final[int] = 12
PLOT_FONTSIZE_TITLE: Final[int] = 14
PLOT_FONTSIZE_LEGEND: Final[int] = 10
PLOT_FONTSIZE_SMALL: Final[int] = 8
PLOT_FONTSIZE_ANNOTATION: Final[int] = 9

# Text offset for bar labels
PLOT_TEXT_OFFSET: Final[float] = 0.02

# X-axis tick rotation (degrees)
PLOT_XTICK_ROTATION: Final[int] = 45

# Figure sizes (width, height in inches)
PLOT_FIGSIZE_CV: Final[Tuple[int, int]] = (10, 6)
PLOT_FIGSIZE_COMPARISON: Final[Tuple[int, int]] = (12, 6)
PLOT_FIGSIZE_CONFUSION: Final[Tuple[int, int]] = (8, 6)
PLOT_FIGSIZE_DETAILED: Final[Tuple[int, int]] = (14, 7)
PLOT_FIGSIZE_SUMMARY: Final[Tuple[int, int]] = (15, 5)
