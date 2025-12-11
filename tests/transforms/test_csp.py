"""
Unit tests for MyCSP class.
"""

import pytest
import numpy as np


class TestMyCSPInit:
    """Tests for MyCSP initialization."""

    def test_init_default_params(self):
        """Test MyCSP initialization with default parameters."""
        from mycsp import MyCSP

        csp = MyCSP()
        assert csp.n_components == 4
        assert csp.reg is None
        assert csp.log is True
        assert csp.norm_trace is True

    def test_init_custom_params(self):
        """Test MyCSP with custom parameters."""
        from mycsp import MyCSP

        csp = MyCSP(n_components=6, reg=0.01, log=False, norm_trace=False)
        assert csp.n_components == 6
        assert csp.reg == 0.01
        assert csp.log is False
        assert csp.norm_trace is False


class TestMyCSPFit:
    """Tests for MyCSP.fit method."""

    def test_fit_returns_self(self, synthetic_eeg_data):
        """Test that fit returns self."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP()
        result = csp.fit(X, y)

        assert result is csp

    def test_fit_creates_attributes(self, synthetic_eeg_data):
        """Test that fit creates W_ and eigenvalues_ attributes."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP(n_components=4)
        csp.fit(X, y)

        assert hasattr(csp, 'W_')
        assert hasattr(csp, 'eigenvalues_')
        assert csp.W_.shape[1] == 4
        assert len(csp.eigenvalues_) == 4

    def test_fit_requires_two_classes(self, synthetic_eeg_3class):
        """Test that fit raises error for non-binary classification."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_3class
        csp = MyCSP()

        with pytest.raises(ValueError, match="CSP requires exactly 2 classes"):
            csp.fit(X, y)

    def test_fit_with_regularization(self, synthetic_eeg_data):
        """Test CSP fit with regularization parameter."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP(n_components=4, reg=0.1)
        csp.fit(X, y)

        assert hasattr(csp, 'W_')
        assert csp.W_.shape[1] == 4

    def test_fit_eigenvalues_sorted(self, synthetic_eeg_data):
        """Test that eigenvalues are properly sorted (first half high, second half low)."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP(n_components=4)
        csp.fit(X, y)

        # First n_components/2 should have higher eigenvalues than last n_components/2
        n_pairs = csp.n_components // 2
        high_eigenvalues = csp.eigenvalues_[:n_pairs]
        low_eigenvalues = csp.eigenvalues_[n_pairs:]

        assert np.mean(high_eigenvalues) > np.mean(low_eigenvalues)


class TestMyCSPTransform:
    """Tests for MyCSP.transform method."""

    def test_transform_not_fitted_raises_error(self, synthetic_eeg_data):
        """Test that transform raises error if not fitted."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP()

        with pytest.raises(RuntimeError, match="CSP not fitted"):
            csp.transform(X)

    def test_transform_output_shape(self, synthetic_eeg_data, n_epochs):
        """Test CSP transform output shape."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        n_components = 4
        csp = MyCSP(n_components=n_components)
        csp.fit(X, y)
        X_csp = csp.transform(X)

        expected_shape = (n_epochs, n_components)
        assert X_csp.shape == expected_shape

    def test_transform_no_nan(self, synthetic_eeg_data):
        """Test that transform does not produce NaN values."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP()
        csp.fit(X, y)
        X_csp = csp.transform(X)

        assert not np.isnan(X_csp).any()

    def test_transform_log_option(self, synthetic_eeg_data):
        """Test log=True vs log=False output."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data

        csp_log = MyCSP(n_components=4, log=True)
        csp_log.fit(X, y)
        csp_log.transform(X)

        csp_no_log = MyCSP(n_components=4, log=False)
        csp_no_log.fit(X, y)
        X_no_log = csp_no_log.transform(X)

        # Log features should be negative or positive (log of values 0-1)
        # Non-log features should be positive (variances)
        assert np.all(X_no_log > 0)

    def test_transform_normalized_variances(self, synthetic_eeg_data):
        """Test that variances are normalized (sum to 1)."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP(n_components=4, log=False)
        csp.fit(X, y)
        X_csp = csp.transform(X)

        # Each epoch's variances should sum to 1
        row_sums = X_csp.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(X_csp)))


class TestMyCSPFitTransform:
    """Tests for MyCSP.fit_transform method."""

    def test_fit_transform_equals_fit_then_transform(self, synthetic_eeg_data):
        """Test that fit_transform equals fit().transform()."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data

        csp1 = MyCSP(n_components=4)
        X_csp1 = csp1.fit_transform(X, y)

        csp2 = MyCSP(n_components=4)
        csp2.fit(X, y)
        X_csp2 = csp2.transform(X)

        # Both should produce same results (with same random state)
        np.testing.assert_array_almost_equal(X_csp1, X_csp2)


class TestMyCSPCovariance:
    """Tests for MyCSP covariance computation."""

    def test_compute_covariance_shape(self, synthetic_eeg_data, n_channels):
        """Test _compute_covariance output shape."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP()
        cov = csp._compute_covariance(X)

        expected_shape = (n_channels, n_channels)
        assert cov.shape == expected_shape

    def test_compute_covariance_symmetric(self, synthetic_eeg_data):
        """Test that covariance matrix is symmetric."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP()
        cov = csp._compute_covariance(X)

        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_compute_covariance_positive_semidefinite(self, synthetic_eeg_data):
        """Test that covariance matrix is positive semi-definite."""
        from mycsp import MyCSP

        X, y = synthetic_eeg_data
        csp = MyCSP()
        cov = csp._compute_covariance(X)

        # All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)  # Small tolerance for numerical errors
