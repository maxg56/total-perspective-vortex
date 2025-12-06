"""
Total Perspective Vortex - EEG Brain-Computer Interface.

A complete BCI system for motor imagery classification using EEG data
from the Physionet EEGMMIDB dataset.

Modules:
- preprocess: Data loading and preprocessing
- features: Feature extraction (PSD, band power)
- mycsp: Custom Common Spatial Patterns implementation
- pipeline: sklearn pipeline construction
- train: Model training and evaluation
- predict: Real-time prediction simulation
- mybci: Main CLI entry point
"""

__version__ = "1.0.0"
__author__ = "Total Perspective Vortex"
