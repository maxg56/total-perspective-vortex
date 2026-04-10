"""
Unit tests for MyNearestCentroid classifier.
"""

import pytest
import numpy as np


class TestMyNearestCentroidInit:
    """Tests for MyNearestCentroid initialization."""

    def test_init_default_params(self):
        """Test default initialization."""
        from classifiers import MyNearestCentroid

        clf = MyNearestCentroid()
        assert clf.shrink_threshold is None

    def test_init_custom_params(self):
        """Test initialization with shrinkage."""
        from classifiers import MyNearestCentroid

        clf = MyNearestCentroid(shrink_threshold=0.5)
        assert clf.shrink_threshold == 0.5


class TestMyNearestCentroidFit:
    """Tests for MyNearestCentroid.fit method."""

    def test_fit_returns_self(self, random_seed):
        """Test that fit returns self."""
        from classifiers import MyNearestCentroid

        X = np.random.randn(20, 5)
        y = np.array([1] * 10 + [2] * 10)
        clf = MyNearestCentroid()
        result = clf.fit(X, y)
        assert result is clf

    def test_fit_creates_attributes(self, random_seed):
        """Test that fit creates necessary attributes."""
        from classifiers import MyNearestCentroid

        X = np.random.randn(20, 5)
        y = np.array([1] * 10 + [2] * 10)
        clf = MyNearestCentroid()
        clf.fit(X, y)

        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'centroids_')
        assert hasattr(clf, 'overall_centroid_')

    def test_fit_classes(self, random_seed):
        """Test that classes are correctly identified."""
        from classifiers import MyNearestCentroid

        X = np.random.randn(30, 5)
        y = np.array([1] * 15 + [2] * 15)
        clf = MyNearestCentroid()
        clf.fit(X, y)

        assert len(clf.classes_) == 2
        assert 1 in clf.classes_
        assert 2 in clf.classes_

    def test_fit_centroids_shape(self, random_seed):
        """Test centroid shapes are correct."""
        from classifiers import MyNearestCentroid

        X = np.random.randn(20, 8)
        y = np.array([1] * 10 + [2] * 10)
        clf = MyNearestCentroid()
        clf.fit(X, y)

        assert clf.centroids_.shape == (2, 8)
        assert clf.overall_centroid_.shape == (8,)

    def test_fit_centroid_values(self, random_seed):
        """Test that centroids are close to expected means."""
        from classifiers import MyNearestCentroid

        n_features = 3
        X_class1 = np.ones((10, n_features)) * 5.0
        X_class2 = np.ones((10, n_features)) * -5.0
        X = np.vstack([X_class1, X_class2])
        y = np.array([1] * 10 + [2] * 10)

        clf = MyNearestCentroid()
        clf.fit(X, y)

        np.testing.assert_array_almost_equal(clf.centroids_[0], [5.0] * 3)
        np.testing.assert_array_almost_equal(clf.centroids_[1], [-5.0] * 3)


class TestMyNearestCentroidPredict:
    """Tests for MyNearestCentroid.predict method."""

    def test_predict_not_fitted_raises_error(self, random_seed):
        """Test that predict raises error if not fitted."""
        from classifiers import MyNearestCentroid

        clf = MyNearestCentroid()
        X = np.random.randn(5, 3)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(X)

    def test_predict_output_shape(self, random_seed):
        """Test prediction output shape."""
        from classifiers import MyNearestCentroid

        X = np.random.randn(20, 5)
        y = np.array([1] * 10 + [2] * 10)
        clf = MyNearestCentroid()
        clf.fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (20,)

    def test_predict_valid_labels(self, random_seed):
        """Test that predictions are valid class labels."""
        from classifiers import MyNearestCentroid

        X = np.random.randn(20, 5)
        y = np.array([1] * 10 + [2] * 10)
        clf = MyNearestCentroid()
        clf.fit(X, y)

        preds = clf.predict(X)
        assert set(preds).issubset({1, 2})

    def test_predict_separable_data(self, random_seed):
        """Test perfect classification on well-separated data."""
        from classifiers import MyNearestCentroid

        X = np.vstack([
            np.random.randn(20, 5) + 5,
            np.random.randn(20, 5) - 5
        ])
        y = np.array([1] * 20 + [2] * 20)

        clf = MyNearestCentroid()
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert acc == 1.0


class TestMyNearestCentroidShrinkage:
    """Tests for shrinkage regularization."""

    def test_shrinkage_moves_centroids_toward_overall(self, random_seed):
        """Test that shrinkage moves centroids toward the overall centroid."""
        from classifiers import MyNearestCentroid

        X = np.vstack([
            np.ones((10, 3)) * 10,
            np.ones((10, 3)) * -10
        ])
        y = np.array([1] * 10 + [2] * 10)

        clf_no_shrink = MyNearestCentroid()
        clf_no_shrink.fit(X, y)

        clf_shrink = MyNearestCentroid(shrink_threshold=5.0)
        clf_shrink.fit(X, y)

        # With shrinkage, centroids should be closer to overall centroid (0)
        dist_no_shrink = np.linalg.norm(
            clf_no_shrink.centroids_ - clf_no_shrink.overall_centroid_)
        dist_shrink = np.linalg.norm(
            clf_shrink.centroids_ - clf_shrink.overall_centroid_)
        assert dist_shrink < dist_no_shrink


class TestMyNearestCentroidScore:
    """Tests for MyNearestCentroid.score method."""

    def test_score_returns_float(self, random_seed):
        """Test that score returns a float."""
        from classifiers import MyNearestCentroid

        X = np.random.randn(20, 5)
        y = np.array([1] * 10 + [2] * 10)
        clf = MyNearestCentroid()
        clf.fit(X, y)

        score = clf.score(X, y)
        assert isinstance(score, float)

    def test_score_between_zero_and_one(self, random_seed):
        """Test that score is between 0 and 1."""
        from classifiers import MyNearestCentroid

        X = np.random.randn(20, 5)
        y = np.array([1] * 10 + [2] * 10)
        clf = MyNearestCentroid()
        clf.fit(X, y)

        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0


class TestMyNearestCentroidSklearnCompat:
    """Tests for sklearn compatibility."""

    def test_sklearn_interface(self):
        """Test MyNearestCentroid has sklearn interface."""
        from classifiers import MyNearestCentroid

        clf = MyNearestCentroid()
        assert hasattr(clf, 'fit')
        assert hasattr(clf, 'predict')
        assert hasattr(clf, 'score')
        assert hasattr(clf, 'get_params')
        assert hasattr(clf, 'set_params')

    def test_get_set_params(self):
        """Test get_params and set_params."""
        from classifiers import MyNearestCentroid

        clf = MyNearestCentroid(shrink_threshold=0.3)
        params = clf.get_params()
        assert params['shrink_threshold'] == 0.3

        clf.set_params(shrink_threshold=0.5)
        assert clf.shrink_threshold == 0.5

    def test_cross_validation(self, random_seed):
        """Test compatibility with sklearn cross_val_score."""
        from classifiers import MyNearestCentroid
        from sklearn.model_selection import cross_val_score

        X = np.vstack([
            np.random.randn(30, 5) + 3,
            np.random.randn(30, 5) - 3
        ])
        y = np.array([1] * 30 + [2] * 30)

        scores = cross_val_score(MyNearestCentroid(), X, y, cv=3)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)
