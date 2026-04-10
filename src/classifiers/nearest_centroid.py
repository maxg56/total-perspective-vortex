"""
Custom Nearest Centroid classifier implementation.

A simple but effective classifier for EEG-based BCI that classifies
samples based on their Euclidean distance to class centroids.
Supports optional shrinkage regularization for high-dimensional data.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin


class MyNearestCentroid(BaseEstimator, ClassifierMixin):
    """
    Custom Nearest Centroid classifier as sklearn-compatible estimator.

    Computes the centroid (mean) of each class during fitting, then
    classifies new samples by assigning them to the class whose
    centroid is closest (Euclidean distance).

    Optionally applies shrinkage regularization which moves class
    centroids toward the overall centroid, helping with
    high-dimensional or noisy features.

    Parameters
    ----------
    shrink_threshold : float or None
        Shrinkage threshold for regularization. If not None, each
        feature's deviation from the overall centroid is shrunk
        toward zero. Higher values produce more regularization.
        Default: None (no shrinkage)

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels found during fit
    centroids_ : np.ndarray
        Class centroids of shape (n_classes, n_features)
    overall_centroid_ : np.ndarray
        Overall centroid of shape (n_features,)
    """

    def __init__(self, shrink_threshold: Optional[float] = None) -> None:
        self.shrink_threshold = shrink_threshold

    def fit(self, X: NDArray[np.float64],
            y: NDArray[np.int64]) -> 'MyNearestCentroid':
        """
        Fit the classifier by computing class centroids.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target labels of shape (n_samples,)

        Returns
        -------
        self : MyNearestCentroid
            The fitted classifier
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Compute overall centroid
        self.overall_centroid_ = np.mean(X, axis=0)

        # Compute per-class centroids from scratch
        self.centroids_ = np.zeros((n_classes, n_features),
                                   dtype=np.float64)
        for i, cls in enumerate(self.classes_):
            mask = (y == cls)
            n_samples_cls: int = int(np.sum(mask))
            # Manual mean computation
            centroid = np.zeros(n_features, dtype=np.float64)
            for j in range(X.shape[0]):
                if mask[j]:
                    for k in range(n_features):
                        centroid[k] += X[j, k]
            centroid /= n_samples_cls
            self.centroids_[i] = centroid

        # Apply shrinkage regularization if threshold is set
        if self.shrink_threshold is not None:
            for i in range(n_classes):
                deviation = self.centroids_[i] - self.overall_centroid_
                # Soft-threshold each feature's deviation
                sign = np.sign(deviation)
                magnitude = np.abs(deviation)
                shrunk = np.maximum(
                    magnitude - self.shrink_threshold, 0.0)
                self.centroids_[i] = (
                    self.overall_centroid_ + sign * shrunk)

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Predict class labels for samples in X.

        Assigns each sample to the class whose centroid is nearest
        (minimum Euclidean distance).

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels of shape (n_samples,)
        """
        if not hasattr(self, 'centroids_'):
            raise RuntimeError(
                "Classifier not fitted. Call fit() first.")

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=self.classes_.dtype)

        for i in range(n_samples):
            # Compute Euclidean distance to each centroid
            min_dist = np.inf
            best_class = self.classes_[0]
            for j, cls in enumerate(self.classes_):
                # Manual distance computation
                dist = 0.0
                for k in range(X.shape[1]):
                    diff = X[i, k] - self.centroids_[j, k]
                    dist += diff * diff
                dist = np.sqrt(dist)

                if dist < min_dist:
                    min_dist = dist
                    best_class = cls

            predictions[i] = best_class

        return predictions

    def score(self, X: NDArray[np.float64],
              y: NDArray[np.int64]) -> float:
        """
        Compute accuracy score for the given test data and labels.

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features)
        y : np.ndarray
            True labels of shape (n_samples,)

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == y))
