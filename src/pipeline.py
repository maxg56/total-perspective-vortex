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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from constants import (EEG_SAMPLING_RATE, DEFAULT_N_COMPONENTS_PCA_PIPELINE,
                       RANDOM_STATE)
from transforms import MyCSP, MyPCA
from features import (PSDExtractor, BandPowerExtractor, FlattenExtractor,
                      WaveletExtractor)
from classifiers import MyNearestCentroid


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


def build_csp_rf_pipeline(n_components: int = 6, reg: Optional[float] = None,
                          n_estimators: int = 100) -> Pipeline:
    """
    Build a CSP + Random Forest pipeline.

    Parameters
    ----------
    n_components : int
        Number of CSP components
    reg : float
        Regularization for CSP
    n_estimators : int
        Number of trees in the forest

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('csp', MyCSP(n_components=n_components, reg=reg, log=True)),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators, random_state=RANDOM_STATE))
    ])
    return pipeline


def build_psd_lda_pipeline(fs: float = EEG_SAMPLING_RATE) -> Pipeline:
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


def build_bandpower_lda_pipeline(fs: float = EEG_SAMPLING_RATE) -> Pipeline:
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


def build_flat_pca_lda_pipeline(n_components: int = DEFAULT_N_COMPONENTS_PCA_PIPELINE) -> Pipeline:
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


def build_wavelet_lda_pipeline(
        fs: float = EEG_SAMPLING_RATE,
        n_scales_per_band: int = 5) -> Pipeline:
    """
    Build a Wavelet + LDA pipeline.

    Uses Continuous Wavelet Transform (Morlet) to extract time-frequency
    energy features, followed by LDA classification.

    Parameters
    ----------
    fs : float
        Sampling frequency
    n_scales_per_band : int
        Number of wavelet scales per frequency band

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('wavelet', WaveletExtractor(
            fs=fs, n_scales_per_band=n_scales_per_band)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    return pipeline


def build_wavelet_custom_pipeline(
        fs: float = EEG_SAMPLING_RATE,
        n_scales_per_band: int = 5,
        n_components: int = 10,
        shrink_threshold: Optional[float] = None) -> Pipeline:
    """
    Build a Wavelet + PCA + custom Nearest Centroid pipeline.

    Uses Continuous Wavelet Transform features with PCA dimensionality
    reduction followed by a custom nearest centroid classifier. PCA is
    necessary because wavelet features live in a high-dimensional space
    (n_channels * n_bands) where nearest-centroid degrades severely
    without prior reduction.

    Parameters
    ----------
    fs : float
        Sampling frequency
    n_scales_per_band : int
        Number of wavelet scales per frequency band
    n_components : int
        Number of PCA components to retain before classification
    shrink_threshold : float or None
        Shrinkage threshold for the classifier

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('wavelet', WaveletExtractor(
            fs=fs, n_scales_per_band=n_scales_per_band)),
        ('scaler', StandardScaler()),
        ('pca', MyPCA(n_components=n_components)),
        ('clf', MyNearestCentroid(
            shrink_threshold=shrink_threshold))
    ])
    return pipeline


def build_csp_custom_pipeline(
        n_components: int = 6,
        reg: Optional[float] = None,
        shrink_threshold: Optional[float] = None) -> Pipeline:
    """
    Build a CSP + custom Nearest Centroid pipeline.

    Uses CSP spatial filtering with a custom nearest centroid classifier.

    Parameters
    ----------
    n_components : int
        Number of CSP components
    reg : float
        Regularization for CSP
    shrink_threshold : float or None
        Shrinkage threshold for the classifier

    Returns
    -------
    pipeline : Pipeline
    """
    pipeline = Pipeline([
        ('csp', MyCSP(n_components=n_components, reg=reg, log=True)),
        ('scaler', StandardScaler()),
        ('clf', MyNearestCentroid(
            shrink_threshold=shrink_threshold))
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
        - 'csp_rf': CSP + Random Forest
        - 'psd_lda': PSD features + LDA
        - 'bandpower_lda': Band power features + LDA
        - 'flat_pca_lda': Flattened + PCA + LDA
        - 'wavelet_lda': Wavelet transform + LDA
        - 'wavelet_custom': Wavelet + custom Nearest Centroid
        - 'csp_custom': CSP + custom Nearest Centroid
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
        'csp_rf': build_csp_rf_pipeline,
        'psd_lda': build_psd_lda_pipeline,
        'bandpower_lda': build_bandpower_lda_pipeline,
        'flat_pca_lda': build_flat_pca_lda_pipeline,
        'wavelet_lda': build_wavelet_lda_pipeline,
        'wavelet_custom': build_wavelet_custom_pipeline,
        'csp_custom': build_csp_custom_pipeline,
    }

    if pipeline_name not in pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. "
                         f"Available: {list(pipelines.keys())}")

    builder = pipelines[pipeline_name]
    return builder(**kwargs)  # type: ignore[operator]


def list_pipelines() -> list:
    """Return list of available pipeline names."""
    return [
        'csp_lda', 'csp_svm', 'csp_lr', 'csp_rf',
        'psd_lda', 'bandpower_lda', 'flat_pca_lda',
        'wavelet_lda', 'wavelet_custom', 'csp_custom',
    ]


if __name__ == "__main__":
    print("Available pipelines:")
    for name in list_pipelines():
        pipeline = get_pipeline(name)
        print(f"\n{name}:")
        for step_name, step in pipeline.steps:
            print(f"  - {step_name}: {step.__class__.__name__}")
