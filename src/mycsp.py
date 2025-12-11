"""
Custom Common Spatial Patterns (CSP) and PCA implementations.

This module provides backwards compatibility. The actual implementations
have been moved to the transforms package:
    - transforms.csp.MyCSP
    - transforms.pca.MyPCA

For new code, prefer importing directly from the transforms package:
    from transforms import MyCSP, MyPCA
"""

# Re-export from transforms package for backwards compatibility
from transforms import MyCSP, MyPCA

__all__ = ['MyCSP', 'MyPCA']


if __name__ == "__main__":
    import numpy as np

    # Test CSP implementation
    print("Testing CSP implementation...")

    # Create dummy EEG data with 2 classes
    np.random.seed(42)
    n_epochs = 100
    n_channels = 64
    n_times = 480

    # Simulate class 1 with higher variance in first channels
    X1 = np.random.randn(n_epochs // 2, n_channels, n_times)
    X1[:, :5, :] *= 2  # Higher variance in first 5 channels

    # Simulate class 2 with higher variance in last channels
    X2 = np.random.randn(n_epochs // 2, n_channels, n_times)
    X2[:, -5:, :] *= 2  # Higher variance in last 5 channels

    X = np.concatenate([X1, X2], axis=0)
    y = np.array([0] * (n_epochs // 2) + [1] * (n_epochs // 2))

    # Shuffle
    perm = np.random.permutation(n_epochs)
    X = X[perm]
    y = y[perm]

    # Test CSP
    csp = MyCSP(n_components=4, log=True)
    X_csp = csp.fit_transform(X, y)

    print(f"Input shape: {X.shape}")
    print(f"CSP output shape: {X_csp.shape}")
    print(f"CSP eigenvalues: {csp.eigenvalues_}")
    print(f"CSP filter shape: {csp.W_.shape}")

    # Test PCA
    X_flat = X.reshape(X.shape[0], -1)
    pca = MyPCA(n_components=10)
    X_pca = pca.fit_transform(X_flat)

    print(f"\nPCA input shape: {X_flat.shape}")
    print(f"PCA output shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
