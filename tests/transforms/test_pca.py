"""
Unit tests for MyPCA class.
"""

import numpy as np


class TestMyPCAInit:
    """Tests for MyPCA initialization."""

    def test_init_default_params(self):
        """Test MyPCA initialization with default parameters."""
        from mycsp import MyPCA

        pca = MyPCA()
        assert pca.n_components == 10

    def test_init_custom_params(self):
        """Test MyPCA with custom parameters."""
        from mycsp import MyPCA

        pca = MyPCA(n_components=20)
        assert pca.n_components == 20


class TestMyPCAFit:
    """Tests for MyPCA.fit method."""

    def test_fit_returns_self(self, flat_2d_data):
        """Test that fit returns self."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        pca = MyPCA()
        result = pca.fit(X)

        assert result is pca

    def test_fit_creates_attributes(self, flat_2d_data):
        """Test that fit creates necessary attributes."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        pca = MyPCA(n_components=5)
        pca.fit(X)

        assert hasattr(pca, 'mean_')
        assert hasattr(pca, 'components_')
        assert hasattr(pca, 'explained_variance_')
        assert hasattr(pca, 'explained_variance_ratio_')
        assert pca.components_.shape[1] == 5

    def test_fit_mean_shape(self, flat_2d_data):
        """Test mean_ has correct shape."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        pca = MyPCA()
        pca.fit(X)

        assert pca.mean_.shape == (X.shape[1],)

    def test_explained_variance_ratio_sums_to_one(self, flat_2d_data):
        """Test that explained variance ratios sum to <= 1."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        pca = MyPCA(n_components=10)
        pca.fit(X)

        # Sum should be <= 1 (we're keeping subset of components)
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-10


class TestMyPCATransform:
    """Tests for MyPCA.transform method."""

    def test_transform_output_shape(self, flat_2d_data):
        """Test PCA transform output shape."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        n_components = 5
        pca = MyPCA(n_components=n_components)
        pca.fit(X)
        X_pca = pca.transform(X)

        expected_shape = (X.shape[0], n_components)
        assert X_pca.shape == expected_shape

    def test_transform_no_nan(self, flat_2d_data):
        """Test that transform does not produce NaN values."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        pca = MyPCA()
        pca.fit(X)
        X_pca = pca.transform(X)

        assert not np.isnan(X_pca).any()

    def test_transform_reduces_dimensionality(self, flat_2d_data):
        """Test that PCA reduces dimensionality."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        n_components = 5
        pca = MyPCA(n_components=n_components)
        pca.fit(X)
        X_pca = pca.transform(X)

        assert X_pca.shape[1] < X.shape[1]
        assert X_pca.shape[1] == n_components


class TestMyPCAConsistency:
    """Tests for MyPCA consistency with sklearn PCA."""

    def test_pca_components_orthogonal(self, flat_2d_data):
        """Test that PCA components are orthogonal."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        pca = MyPCA(n_components=5)
        pca.fit(X)

        # Components should be orthogonal (dot product ≈ 0 for different components)
        dot_product = np.dot(pca.components_.T, pca.components_)
        np.testing.assert_array_almost_equal(dot_product, np.eye(5), decimal=10)

    def test_explained_variance_descending(self, flat_2d_data):
        """Test that explained variance is in descending order."""
        from mycsp import MyPCA

        X, y = flat_2d_data
        pca = MyPCA(n_components=10)
        pca.fit(X)

        # Each variance should be >= next variance
        for i in range(len(pca.explained_variance_) - 1):
            assert pca.explained_variance_[i] >= pca.explained_variance_[i + 1]
