"""
Tests for new visualization functions.

Covers: plot_csp_filters, plot_learning_curve, plot_class_metrics, plot_roc_curve.
"""

import os
import tempfile
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from visualization.csp_plots import plot_csp_filters
from visualization.learning_curve_plots import plot_learning_curve
from visualization.advanced_metrics_plots import plot_class_metrics, plot_roc_curve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_labels():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=60)
    return y


@pytest.fixture
def binary_predictions(binary_labels):
    rng = np.random.default_rng(1)
    # Flip ~20 % of labels to introduce errors
    flip = rng.random(size=len(binary_labels)) < 0.2
    y_pred = binary_labels.copy()
    y_pred[flip] = 1 - y_pred[flip]
    return y_pred


@pytest.fixture
def decision_scores(binary_labels):
    rng = np.random.default_rng(2)
    # Scores correlated with true label (better than random)
    return binary_labels.astype(float) + rng.normal(0, 0.5, size=len(binary_labels))


@pytest.fixture
def csp_filters():
    rng = np.random.default_rng(3)
    n_channels, n_components = 64, 4
    W = rng.standard_normal((n_channels, n_components))
    eigenvalues = np.array([0.85, 0.70, 0.30, 0.15])
    return W, eigenvalues


@pytest.fixture
def simple_pipeline():
    """Return a fresh unfitted CSP+LDA pipeline."""
    from pipeline import get_pipeline
    return get_pipeline('csp_lda')


@pytest.fixture
def separable_eeg():
    """Small separable 2-class EEG dataset for learning curve tests."""
    rng = np.random.default_rng(4)
    n_epochs, n_channels, n_times = 40, 8, 80
    X = rng.standard_normal((n_epochs, n_channels, n_times))
    y = np.array([0] * (n_epochs // 2) + [1] * (n_epochs // 2))
    X[:n_epochs // 2, :4, :] += 2.0
    return X, y


# ---------------------------------------------------------------------------
# plot_csp_filters
# ---------------------------------------------------------------------------

class TestPlotCspFilters:
    def test_returns_figure(self, csp_filters):
        W, eigenvalues = csp_filters
        fig = plot_csp_filters(W, eigenvalues)
        assert isinstance(fig, plt.Figure)

    def test_with_channel_names(self, csp_filters):
        W, eigenvalues = csp_filters
        ch_names = [f'Ch{i}' for i in range(W.shape[0])]
        fig = plot_csp_filters(W, eigenvalues, channel_names=ch_names)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, csp_filters):
        W, eigenvalues = csp_filters
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            plot_csp_filters(W, eigenvalues, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# plot_learning_curve
# ---------------------------------------------------------------------------

class TestPlotLearningCurve:
    def test_returns_figure(self, simple_pipeline, separable_eeg):
        X, y = separable_eeg
        fig = plot_learning_curve(simple_pipeline, X, y, cv=2)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, simple_pipeline, separable_eeg):
        X, y = separable_eeg
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            plot_learning_curve(simple_pipeline, X, y, cv=2, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# plot_class_metrics
# ---------------------------------------------------------------------------

class TestPlotClassMetrics:
    def test_returns_figure(self, binary_labels, binary_predictions):
        fig = plot_class_metrics(binary_labels, binary_predictions)
        assert isinstance(fig, plt.Figure)

    def test_with_class_names(self, binary_labels, binary_predictions):
        fig = plot_class_metrics(
            binary_labels, binary_predictions,
            class_names=['Left', 'Right']
        )
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, binary_labels, binary_predictions):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            plot_class_metrics(binary_labels, binary_predictions, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------

class TestPlotRocCurve:
    def test_returns_figure(self, binary_labels, decision_scores):
        fig = plot_roc_curve(binary_labels, decision_scores)
        assert isinstance(fig, plt.Figure)

    def test_with_class_names(self, binary_labels, decision_scores):
        fig = plot_roc_curve(
            binary_labels, decision_scores,
            class_names=['Left', 'Right']
        )
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, binary_labels, decision_scores):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            plot_roc_curve(binary_labels, decision_scores, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
