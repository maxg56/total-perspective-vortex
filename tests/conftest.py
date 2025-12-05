"""
Pytest configuration and fixtures for EEG BCI tests.

Provides synthetic EEG data to avoid downloading from Physionet during testing.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def random_seed():
    """Fix random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def n_epochs():
    """Standard number of epochs for testing."""
    return 50


@pytest.fixture
def n_channels():
    """Standard number of EEG channels (Physionet has 64)."""
    return 64


@pytest.fixture
def n_times():
    """Standard number of time points (3 seconds at 160 Hz)."""
    return 480


@pytest.fixture
def fs():
    """Standard sampling frequency."""
    return 160.0


@pytest.fixture
def synthetic_eeg_data(random_seed, n_epochs, n_channels, n_times):
    """
    Generate synthetic EEG data for testing.

    Creates two classes with different spatial patterns to simulate
    motor imagery data.

    Returns
    -------
    X : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times)
    y : np.ndarray
        Labels of shape (n_epochs,) with values 1 and 2 (matching Physionet)
    """
    np.random.seed(42)

    # Generate base noise
    X = np.random.randn(n_epochs, n_channels, n_times).astype(np.float64)

    # Create labels (half class 1, half class 2)
    y = np.array([1] * (n_epochs // 2) + [2] * (n_epochs // 2))

    # Add class-specific patterns
    # Class 1: Higher variance in first 10 channels (simulating left motor cortex)
    X[:n_epochs // 2, :10, :] *= 2.0

    # Class 2: Higher variance in channels 10-20 (simulating right motor cortex)
    X[n_epochs // 2:, 10:20, :] *= 2.0

    # Shuffle data
    perm = np.random.permutation(n_epochs)
    X = X[perm]
    y = y[perm]

    return X, y


@pytest.fixture
def synthetic_eeg_3class(random_seed, n_epochs, n_channels, n_times):
    """
    Generate synthetic 3-class EEG data for testing error handling.

    Returns
    -------
    X : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times)
    y : np.ndarray
        Labels with 3 classes
    """
    np.random.seed(42)

    n_per_class = n_epochs // 3
    X = np.random.randn(n_per_class * 3, n_channels, n_times)
    y = np.array([1] * n_per_class + [2] * n_per_class + [3] * n_per_class)

    return X, y


@pytest.fixture
def small_synthetic_data(random_seed):
    """
    Generate small synthetic data for quick tests.

    Returns
    -------
    X : np.ndarray
        Small EEG data of shape (20, 16, 160)
    y : np.ndarray
        Labels
    """
    np.random.seed(42)

    n_epochs, n_channels, n_times = 20, 16, 160
    X = np.random.randn(n_epochs, n_channels, n_times)
    y = np.array([1] * 10 + [2] * 10)

    # Add class-specific variance
    X[:10, :4, :] *= 2.0
    X[10:, 4:8, :] *= 2.0

    perm = np.random.permutation(n_epochs)
    X = X[perm]
    y = y[perm]

    return X, y


@pytest.fixture
def flat_2d_data(random_seed):
    """
    Generate flat 2D data for PCA testing.

    Returns
    -------
    X : np.ndarray
        2D data of shape (n_samples, n_features)
    y : np.ndarray
        Labels
    """
    np.random.seed(42)

    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)
    y = np.array([0] * 50 + [1] * 50)

    return X, y


@pytest.fixture
def single_epoch(n_channels, n_times):
    """Generate a single epoch for prediction testing."""
    np.random.seed(42)
    return np.random.randn(n_channels, n_times)


@pytest.fixture
def trained_csp_pipeline(small_synthetic_data):
    """
    Create a trained CSP+LDA pipeline for testing.

    Returns
    -------
    pipeline : Pipeline
        Fitted sklearn pipeline
    X : np.ndarray
        Training data
    y : np.ndarray
        Training labels
    """
    from pipeline import get_pipeline

    X, y = small_synthetic_data
    pipeline = get_pipeline('csp_lda', n_components=4)
    pipeline.fit(X, y)

    return pipeline, X, y


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)
