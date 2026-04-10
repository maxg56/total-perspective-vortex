# Total Perspective Vortex

EEG Brain-Computer Interface (BCI) system for motor imagery classification using the Physionet EEGMMIDB dataset.

## Overview

This project implements a complete BCI pipeline:
1. **Preprocessing**: Load and filter EEG data from Physionet
2. **Feature Extraction**: PSD, band power, CSP, or wavelet features
3. **Classification**: LDA, SVM, Logistic Regression, Random Forest, or Nearest Centroid
4. **Real-time Prediction**: Simulate streaming EEG classification

## Methodological Choices

This section documents the rationale behind key hyperparameters and design decisions in the recommended `csp_lda` pipeline.

### 7–30 Hz Frequency Band

The bandpass filter (defined in [constants.py](src/constants.py)) targets the **mu (8–12 Hz)** and **beta (12–30 Hz)** frequency bands, which are the primary neural oscillations modulated during motor imagery and motor execution tasks. These bands exhibit event-related desynchronization (ERD) and synchronization (ERS) patterns that are critical for discriminating between different motor imagery conditions.

**Oral explanation**: *We filter EEG signals between 7-30 Hz to capture mu and beta rhythms, which are modulated during motor imagery. Lower frequencies (delta, theta) and higher frequencies (gamma) are excluded to reduce artifacts and focus on motor-related brain activity.*

### 6 CSP Components

The CSP algorithm (implemented in [transforms/csp.py](src/transforms/csp.py)) selects `n_components/2` spatial filters from each end of the eigenvalue spectrum—with 6 components, we extract **3 filters optimized for each class**. This configuration balances discriminative power (capturing class-specific spatial patterns) with dimensionality reduction to prevent overfitting, particularly given the limited number of trials (~45 epochs) per run in the Physionet dataset.

**Oral explanation**: *Six CSP components (3 per class) provide sufficient discriminative information while avoiding overfitting on small datasets. More components would increase model complexity without proportional gains in accuracy.*

### Log-Variance Feature Extraction

CSP features are computed as the **log-variance of spatially filtered signals** ([transforms/csp.py](src/transforms/csp.py)). The logarithm transforms the multiplicative variance structure into an additive one, making the feature distribution approximately Gaussian—a critical assumption for optimal Linear Discriminant Analysis performance.

**Oral explanation**: *Log-variance transformation normalizes CSP feature distributions to be approximately Gaussian. This mathematical property directly aligns with LDA's assumption of normally distributed features, maximizing classifier performance.*

### LDA Classifier

Linear Discriminant Analysis ([pipeline.py](src/pipeline.py)) is the standard classifier for CSP-based BCIs because it assumes **Gaussian class-conditional distributions**—precisely what log-variance CSP features provide. LDA is computationally efficient, requires no hyperparameter tuning, and finds the optimal linear decision boundary when its Gaussian assumption holds, making it ideal for real-time BCI applications.

**Oral explanation**: *LDA is optimal for Gaussian-distributed features like log-variance CSP outputs. It requires minimal computation and no hyperparameter tuning, making it suitable for real-time brain-computer interfaces.*

## Installation

```bash
uv sync
```

## Usage

### Training

Train a model on a specific subject and run:

```bash
cd src
uv run python mybci.py <subject> <run> train
```

Example:
```bash
uv run python mybci.py 4 14 train
```

### Prediction

Run prediction with a trained model:

```bash
uv run python mybci.py <subject> <run> predict
```

Example:
```bash
uv run python mybci.py 4 14 predict
```

### Options

```
--pipeline, -p     Pipeline to use (see available pipelines below)
--cv               Number of cross-validation folds (default: 5)
--compare          Compare all pipelines and use the best
--n-components     Number of CSP/PCA components (default: 6)
--model-dir        Directory for saved models (default: models)
```

### Available Pipelines

| Name | Description |
|------|-------------|
| `csp_lda` | CSP + LDA (recommended, best for motor imagery) |
| `csp_svm` | CSP + SVM |
| `csp_lr` | CSP + Logistic Regression |
| `csp_rf` | CSP + Random Forest |
| `psd_lda` | Power Spectral Density features + LDA |
| `bandpower_lda` | Band power features + LDA |
| `flat_pca_lda` | Flattened raw signal + PCA + LDA |
| `wavelet_lda` | Wavelet (CWT Morlet) features + LDA |
| `wavelet_custom` | Wavelet + PCA + MyNearestCentroid |
| `csp_custom` | CSP + MyNearestCentroid |

### Visualization

Training and prediction commands support optional plot generation:

```bash
# Save plots to disk during training
cd src
uv run python mybci.py 4 14 train --save-plots

# Compare all pipelines with plots
uv run python mybci.py 4 14 train --compare --save-plots
```

Available visualizations:
- **Cross-validation scores**: Bar chart showing accuracy for each CV fold
- **Confusion matrix**: Heatmap of prediction results on test set
- **Training summary**: Comprehensive 3-panel view with CV scores, confusion matrix, and per-class accuracy
- **Pipeline comparison**: Compare performance of different models (when using `--compare`)

Plots are saved to the `plots/` directory.

## Run Types

| Run Numbers | Task Type |
|-------------|-----------|
| 3, 7, 11    | Motor execution (left fist vs right fist) |
| 4, 8, 12    | Motor imagery (left fist vs right fist) |
| 5, 9, 13    | Motor execution (both fists vs both feet) |
| 6, 10, 14   | Motor imagery (both fists vs both feet) |

## Project Structure

```
src/
├── mybci.py              # Main CLI entry point
├── preprocess.py         # Data loading and bandpass filtering (7-30 Hz)
├── features.py           # Feature extractors (PSD, BandPower, Flatten, Wavelet)
├── display.py            # Centralized terminal output
├── pipeline.py           # Pipeline construction and registration
├── predict.py            # Real-time prediction simulation
├── constants.py          # Centralized configuration and magic numbers
├── transforms/
│   ├── csp.py            # MyCSP (solves generalized eigenvalue problem)
│   ├── pca.py            # MyPCA implementation
│   └── linalg.py         # Custom linear algebra (Jacobi eigendecomposition)
├── classifiers/
│   └── nearest_centroid.py  # MyNearestCentroid classifier
├── training/
│   ├── core.py           # Cross-validation and holdout evaluation
│   ├── subject.py        # Subject-specific training logic
│   ├── comparison.py     # Multi-pipeline comparison
│   └── persistence.py    # Model saving/loading with joblib
└── visualization/
    ├── cv_plots.py        # Cross-validation score plots
    ├── metrics_plots.py   # Confusion matrices and classification reports
    └── comparison_plots.py  # Pipeline comparison charts

models/             # Saved models (model_s{subject}_r{run}_{pipeline}.pkl)
plots/              # Generated visualization plots
```

## Testing

The project includes a comprehensive test suite with 270 tests using pytest.

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/transforms/test_csp.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

| Test directory / file | Description |
|-----------------------|-------------|
| `test_preprocess.py` | Run type identification, event IDs, parameter validation |
| `features/` | PSD, BandPower, LogVariance, Flatten, Wavelet extractors |
| `transforms/` | CSP fitting/transform, PCA, linalg (Jacobi), sklearn compat |
| `pipeline/` | All 10 pipeline configurations, factory, integration |
| `classifiers/` | MyNearestCentroid fit/predict/sklearn compat |
| `test_train.py` | Cross-validation, holdout training, model save/load |
| `test_predict.py` | Single/batch prediction, real-time simulation |
| `test_display.py` | All terminal output functions |

### Synthetic Data

Tests use synthetic EEG data to avoid downloading from Physionet during testing. The fixtures in `conftest.py` generate:
- 2-class EEG data with different spatial patterns
- Configurable dimensions (epochs, channels, time points)
- Pre-trained pipelines for prediction tests

## Custom Implementations

### Common Spatial Patterns (CSP)

[transforms/csp.py](src/transforms/csp.py) implements CSP as an sklearn transformer:
- Computes class-specific covariance matrices
- Solves the generalized eigenvalue problem `Σ₁W = (Σ₁ + Σ₂)WΛ`
- Selects `n_components/2` filters from each end of the eigenvalue spectrum
- Returns log-variance features for classification

### Principal Component Analysis (PCA)

[transforms/pca.py](src/transforms/pca.py) implements PCA from scratch using the custom Jacobi eigendecomposition from [transforms/linalg.py](src/transforms/linalg.py).

### Nearest Centroid Classifier

[classifiers/nearest_centroid.py](src/classifiers/nearest_centroid.py) implements a nearest centroid classifier with optional shrinkage threshold, used in `wavelet_custom` and `csp_custom` pipelines.

## Target Performance

- **Minimum accuracy**: 60% on test data
- **Prediction time**: < 2 seconds per epoch

## Manual Validation

```bash
# Subject 1, Run 6 (hands vs feet motor imagery)
cd src
uv run python mybci.py 1 6 train --cv 5
uv run python mybci.py 1 6 predict

# Subject 4, Run 14 (hands vs feet motor imagery)
uv run python mybci.py 4 14 train --cv 5
uv run python mybci.py 4 14 predict

# Subject 10, Run 6 (hands vs feet motor imagery)
uv run python mybci.py 10 6 train --cv 5
uv run python mybci.py 10 6 predict
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd total-perspective-vortex

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality Tools

- **flake8**: Style guide enforcement (PEP 8, max line length 100)
- **mypy**: Static type checking
- **pre-commit**: Automatic checks before commits

### Running Code Quality Checks

```bash
# Run flake8
uv run flake8 src/ tests/

# Run mypy for type checking
uv run mypy src/ --ignore-missing-imports

# Run all pre-commit hooks manually
uv run pre-commit run --all-files

# Run tests
uv run pytest tests/ -v
```

### Configuration Files

- `.flake8`: Flake8 configuration (line length, ignored errors)
- `setup.cfg`: Mypy configuration
- `.pre-commit-config.yaml`: Pre-commit hooks configuration

## License

MIT License
