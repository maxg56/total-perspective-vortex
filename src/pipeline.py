"""
Pipeline construction module for EEG classification.

Builds sklearn pipelines combining:
- Feature extraction (CSP, PSD, etc.)
- Dimensionality reduction (custom CSP/PCA)
- Classification (LDA, SVM, etc.)
"""

from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mycsp import MyCSP, MyPCA
from features import PSDExtractor, BandPowerExtractor, FlattenExtractor


def build_csp_lda_pipeline(n_components: int = 6, reg: Optional[float] = None) -> Pipeline:
    """
    Build a CSP + LDA pipeline (recommended for motor imagery).

    This is the standard pipeline for EEG-based BCI:
    1. CSP spatial filtering (extracts discriminative spatial patterns)
    2. LDA classification (linear classifier optimal for CSP features)

    Parameters
    ----------
    n_components : int
        Number of CSP components (default: 6)
    reg : float
        Regularization for CSP (default: None)

    Returns
    -------
    pipeline : Pipeline
        sklearn Pipeline object
    """
    pipeline = Pipeline([
        ('csp', MyCSP(n_components=n_components, reg=reg, log=True)),
        ('lda', LinearDiscriminantAnalysis())
    ])
    return pipeline


def build_csp_svm_pipeline(n_components: int = 6, reg: Optional[float] = None,
                           C: float = 1.0, kernel: str = 'rbf') -> Pipeline:
    """
    Build a CSP + SVM pipeline.

    Parameters
    ----------
    n_components : int
        Number of CSP components
    reg : float
        Regularization for CSP
    C : float
        SVM regularization parameter
    kernel : str
        SVM kernel type

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('csp', MyCSP(n_components=n_components, reg=reg, log=True)),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=C, kernel=kernel))
    ])
    return pipeline


def build_csp_lr_pipeline(n_components: int = 6, reg: Optional[float] = None,
                          C: float = 1.0) -> Pipeline:
    """
    Build a CSP + Logistic Regression pipeline.

    Parameters
    ----------
    n_components : int
        Number of CSP components
    reg : float
        Regularization for CSP
    C : float
        Logistic regression regularization parameter

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('csp', MyCSP(n_components=n_components, reg=reg, log=True)),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=C, max_iter=1000))
    ])
    return pipeline


def build_psd_lda_pipeline(fs: float = 160.0) -> Pipeline:
    """
    Build a PSD + LDA pipeline.

    Alternative to CSP, uses power spectral features.

    Parameters
    ----------
    fs : float
        Sampling frequency

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('psd', PSDExtractor(fs=fs)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    return pipeline


def build_bandpower_lda_pipeline(fs: float = 160.0) -> Pipeline:
    """
    Build a Band Power + LDA pipeline.

    Parameters
    ----------
    fs : float
        Sampling frequency

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('bandpower', BandPowerExtractor(fs=fs)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    return pipeline


def build_flat_pca_lda_pipeline(n_components: int = 50) -> Pipeline:
    """
    Build a Flatten + PCA + LDA pipeline.

    Uses raw signal features with PCA dimensionality reduction.

    Parameters
    ----------
    n_components : int
        Number of PCA components

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('flatten', FlattenExtractor()),
        ('scaler', StandardScaler()),
        ('pca', MyPCA(n_components=n_components)),
        ('lda', LinearDiscriminantAnalysis())
    ])
    return pipeline


def get_pipeline(pipeline_name: str = 'csp_lda', **kwargs) -> Pipeline:
    """
    Get a pipeline by name.

    Parameters
    ----------
    pipeline_name : str
        Name of the pipeline:
        - 'csp_lda': CSP + LDA (default, recommended)
        - 'csp_svm': CSP + SVM
        - 'csp_lr': CSP + Logistic Regression
        - 'psd_lda': PSD features + LDA
        - 'bandpower_lda': Band power features + LDA
        - 'flat_pca_lda': Flattened + PCA + LDA
    **kwargs : dict
        Additional arguments for the pipeline constructor

    Returns
    -------
    pipeline : Pipeline
    """
    pipelines = {
        'csp_lda': build_csp_lda_pipeline,
        'csp_svm': build_csp_svm_pipeline,
        'csp_lr': build_csp_lr_pipeline,
        'psd_lda': build_psd_lda_pipeline,
        'bandpower_lda': build_bandpower_lda_pipeline,
        'flat_pca_lda': build_flat_pca_lda_pipeline,
    }

    if pipeline_name not in pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. "
                         f"Available: {list(pipelines.keys())}")

    builder = pipelines[pipeline_name]
    return builder(**kwargs)  # type: ignore[operator]


def list_pipelines() -> list:
    """Return list of available pipeline names."""
    return ['csp_lda', 'csp_svm', 'csp_lr', 'psd_lda', 'bandpower_lda', 'flat_pca_lda']


if __name__ == "__main__":
    print("Available pipelines:")
    for name in list_pipelines():
        pipeline = get_pipeline(name)
        print(f"\n{name}:")
        for step_name, step in pipeline.steps:
            print(f"  - {step_name}: {step.__class__.__name__}")
