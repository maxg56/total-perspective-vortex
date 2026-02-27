"""
Unit tests for custom linear algebra module.
"""

import pytest
import numpy as np


class TestMyCovariance:
    """Tests for my_covariance function."""

    def test_covariance_shape(self, random_seed):
        """Test output shape of covariance matrix."""
        from transforms.linalg import my_covariance

        X = np.random.randn(5, 20)
        cov = my_covariance(X, rowvar=True)
        assert cov.shape == (5, 5)

    def test_covariance_symmetric(self, random_seed):
        """Test that covariance matrix is symmetric."""
        from transforms.linalg import my_covariance

        X = np.random.randn(10, 50)
        cov = my_covariance(X, rowvar=True)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_covariance_matches_numpy(self, random_seed):
        """Test that custom covariance matches numpy.cov."""
        from transforms.linalg import my_covariance

        X = np.random.randn(5, 20)
        cov_custom = my_covariance(X, rowvar=True)
        cov_numpy = np.cov(X)
        np.testing.assert_array_almost_equal(cov_custom, cov_numpy, decimal=10)

    def test_covariance_rowvar_false(self, random_seed):
        """Test covariance with rowvar=False (columns are variables)."""
        from transforms.linalg import my_covariance

        X = np.random.randn(20, 5)
        cov_custom = my_covariance(X, rowvar=False)
        cov_numpy = np.cov(X, rowvar=False)
        np.testing.assert_array_almost_equal(cov_custom, cov_numpy, decimal=10)

    def test_covariance_positive_semidefinite(self, random_seed):
        """Test that covariance matrix is positive semi-definite."""
        from transforms.linalg import my_covariance

        X = np.random.randn(5, 30)
        cov = my_covariance(X, rowvar=True)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)


class TestMyCholesky:
    """Tests for my_cholesky function."""

    def test_cholesky_reconstruction(self, random_seed):
        """Test that L @ L.T reconstructs the original matrix."""
        from transforms.linalg import my_cholesky

        A = np.random.randn(5, 5)
        A = A @ A.T + 0.1 * np.eye(5)  # positive definite
        L = my_cholesky(A)
        np.testing.assert_array_almost_equal(L @ L.T, A)

    def test_cholesky_lower_triangular(self, random_seed):
        """Test that L is lower triangular."""
        from transforms.linalg import my_cholesky

        A = np.random.randn(4, 4)
        A = A @ A.T + np.eye(4)
        L = my_cholesky(A)

        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(L[i, j]) < 1e-15

    def test_cholesky_positive_diagonal(self, random_seed):
        """Test that diagonal elements of L are positive."""
        from transforms.linalg import my_cholesky

        A = np.random.randn(6, 6)
        A = A @ A.T + np.eye(6)
        L = my_cholesky(A)
        assert np.all(np.diag(L) > 0)

    def test_cholesky_not_positive_definite(self):
        """Test that non-positive-definite matrix raises ValueError."""
        from transforms.linalg import my_cholesky

        A = np.array([[-1, 0], [0, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="not positive definite"):
            my_cholesky(A)


class TestMyEigh:
    """Tests for my_eigh Jacobi eigenvalue algorithm."""

    def test_eigh_eigenvalues_correct(self, random_seed):
        """Test that eigenvalues satisfy A @ v = lambda * v."""
        from transforms.linalg import my_eigh

        A = np.random.randn(6, 6)
        A = A @ A.T  # symmetric
        eigenvalues, eigenvectors = my_eigh(A)

        for i in range(6):
            residual = A @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i]
            assert np.linalg.norm(residual) < 1e-6

    def test_eigh_ascending_order(self, random_seed):
        """Test that eigenvalues are in ascending order."""
        from transforms.linalg import my_eigh

        A = np.random.randn(8, 8)
        A = A @ A.T
        eigenvalues, _ = my_eigh(A)

        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] <= eigenvalues[i + 1] + 1e-10

    def test_eigh_orthogonal_eigenvectors(self, random_seed):
        """Test that eigenvectors are orthogonal."""
        from transforms.linalg import my_eigh

        A = np.random.randn(5, 5)
        A = A @ A.T
        _, eigenvectors = my_eigh(A)

        dot_product = eigenvectors.T @ eigenvectors
        np.testing.assert_array_almost_equal(dot_product, np.eye(5), decimal=6)

    def test_eigh_matches_numpy(self, random_seed):
        """Test that eigenvalues match numpy's implementation."""
        from transforms.linalg import my_eigh

        A = np.random.randn(6, 6)
        A = A @ A.T
        eigenvalues_custom, _ = my_eigh(A)
        eigenvalues_numpy = np.linalg.eigvalsh(A)

        np.testing.assert_array_almost_equal(
            eigenvalues_custom, eigenvalues_numpy, decimal=6)

    def test_eigh_diagonal_matrix(self):
        """Test eigendecomposition of a diagonal matrix."""
        from transforms.linalg import my_eigh

        A = np.diag([3.0, 1.0, 4.0, 1.5])
        eigenvalues, _ = my_eigh(A)
        expected = np.array([1.0, 1.5, 3.0, 4.0])
        np.testing.assert_array_almost_equal(eigenvalues, expected)

    def test_eigh_identity_matrix(self):
        """Test eigendecomposition of identity matrix."""
        from transforms.linalg import my_eigh

        A = np.eye(4)
        eigenvalues, eigenvectors = my_eigh(A)
        np.testing.assert_array_almost_equal(eigenvalues, np.ones(4))


class TestMyEighGeneralized:
    """Tests for my_eigh_generalized function."""

    def test_generalized_eigenvalues_correct(self, random_seed):
        """Test that A @ v = lambda * B @ v."""
        from transforms.linalg import my_eigh_generalized

        A = np.random.randn(5, 5)
        A = A @ A.T
        B = np.random.randn(5, 5)
        B = B @ B.T + 0.1 * np.eye(5)

        eigenvalues, eigenvectors = my_eigh_generalized(A, B)

        for i in range(5):
            lhs = A @ eigenvectors[:, i]
            rhs = eigenvalues[i] * B @ eigenvectors[:, i]
            if np.linalg.norm(eigenvectors[:, i]) > 0.01:
                residual = np.linalg.norm(lhs - rhs)
                scale = np.linalg.norm(lhs) + 1e-10
                assert residual / scale < 0.05

    def test_generalized_ascending_order(self, random_seed):
        """Test that generalized eigenvalues are in ascending order."""
        from transforms.linalg import my_eigh_generalized

        A = np.random.randn(4, 4)
        A = A @ A.T
        B = np.random.randn(4, 4)
        B = B @ B.T + 0.5 * np.eye(4)

        eigenvalues, _ = my_eigh_generalized(A, B)

        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] <= eigenvalues[i + 1] + 1e-6


class TestMySVD:
    """Tests for my_svd function."""

    def test_svd_reconstruction(self, random_seed):
        """Test that U @ diag(s) @ Vt reconstructs A."""
        from transforms.linalg import my_svd

        A = np.random.randn(5, 3)
        U, s, Vt = my_svd(A)
        A_reconstructed = U @ np.diag(s) @ Vt
        np.testing.assert_array_almost_equal(A, A_reconstructed, decimal=6)

    def test_svd_singular_values_nonnegative(self, random_seed):
        """Test that singular values are non-negative."""
        from transforms.linalg import my_svd

        A = np.random.randn(4, 6)
        _, s, _ = my_svd(A)
        assert np.all(s >= -1e-10)

    def test_svd_singular_values_descending(self, random_seed):
        """Test that singular values are in descending order."""
        from transforms.linalg import my_svd

        A = np.random.randn(5, 4)
        _, s, _ = my_svd(A)

        for i in range(len(s) - 1):
            assert s[i] >= s[i + 1] - 1e-10

    def test_svd_orthogonal_U(self, random_seed):
        """Test that U has orthonormal columns."""
        from transforms.linalg import my_svd

        A = np.random.randn(6, 4)
        U, _, _ = my_svd(A)
        dot = U.T @ U
        np.testing.assert_array_almost_equal(dot, np.eye(4), decimal=6)

    def test_svd_orthogonal_Vt(self, random_seed):
        """Test that Vt has orthonormal rows."""
        from transforms.linalg import my_svd

        A = np.random.randn(6, 4)
        _, _, Vt = my_svd(A)
        dot = Vt @ Vt.T
        np.testing.assert_array_almost_equal(dot, np.eye(4), decimal=6)

    def test_svd_tall_matrix(self, random_seed):
        """Test SVD with m > n."""
        from transforms.linalg import my_svd

        A = np.random.randn(8, 3)
        U, s, Vt = my_svd(A)
        assert U.shape == (8, 3)
        assert s.shape == (3,)
        assert Vt.shape == (3, 3)

    def test_svd_wide_matrix(self, random_seed):
        """Test SVD with m < n (dual trick path)."""
        from transforms.linalg import my_svd

        A = np.random.randn(3, 8)
        U, s, Vt = my_svd(A)
        assert U.shape == (3, 3)
        assert s.shape == (3,)
        assert Vt.shape == (3, 8)

        A_reconstructed = U @ np.diag(s) @ Vt
        np.testing.assert_array_almost_equal(A, A_reconstructed, decimal=6)
