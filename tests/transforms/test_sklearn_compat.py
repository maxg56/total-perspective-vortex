"""
Tests for sklearn transformer interface compatibility of transforms.
"""


class TestSklearnCompatibility:
    """Tests for sklearn transformer interface compatibility."""

    def test_csp_sklearn_interface(self):
        """Test MyCSP is sklearn compatible."""
        from mycsp import MyCSP

        csp = MyCSP()
        assert hasattr(csp, 'fit')
        assert hasattr(csp, 'transform')
        assert hasattr(csp, 'fit_transform')
        assert hasattr(csp, 'get_params')
        assert hasattr(csp, 'set_params')

    def test_pca_sklearn_interface(self):
        """Test MyPCA is sklearn compatible."""
        from mycsp import MyPCA

        pca = MyPCA()
        assert hasattr(pca, 'fit')
        assert hasattr(pca, 'transform')
        assert hasattr(pca, 'get_params')
        assert hasattr(pca, 'set_params')

    def test_csp_get_set_params(self):
        """Test CSP get_params and set_params."""
        from mycsp import MyCSP

        csp = MyCSP(n_components=4, reg=0.1)
        params = csp.get_params()

        assert params['n_components'] == 4
        assert params['reg'] == 0.1

        csp.set_params(n_components=6)
        assert csp.n_components == 6

    def test_pca_get_set_params(self):
        """Test PCA get_params and set_params."""
        from mycsp import MyPCA

        pca = MyPCA(n_components=10)
        params = pca.get_params()

        assert params['n_components'] == 10

        pca.set_params(n_components=20)
        assert pca.n_components == 20
