# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Total Perspective Vortex is an EEG Brain-Computer Interface (BCI) system for motor imagery classification using the Physionet EEGMMIDB dataset. It implements a complete BCI pipeline from preprocessing to real-time prediction.

## Development Commands

### Testing
```bash
# Run all tests (303 tests total)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_mycsp.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Run flake8 style checking
uv run flake8 src/ tests/

# Run mypy type checking
uv run mypy src/ --ignore-missing-imports

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

### Training and Prediction
```bash
# Train a model (from project root)
uv run python mybci.py <subject> <run> train [--pipeline csp_lda] [--cv 5]

# Run prediction
uv run python mybci.py <subject> <run> predict

# Compare all pipelines
uv run python mybci.py 4 14 train --compare
```

### CLI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--pipeline` | `-p` | `csp_lda` | Pipeline to use |
| `--cv` | | `5` | Number of cross-validation folds |
| `--compare` | | off | Compare all pipelines |
| `--model-dir` | | `models` | Directory for saved models |
| `--model-path` | | — | Specific model path (predict mode) |
| `--n-components` | | `6` | Number of CSP components |
| `--reg` | | `None` | CSP regularization parameter |
| `--quiet` | `-q` | off | Reduce output verbosity |
| `--no-plot` | | off | Disable all plot generation |
| `--save-plots` | | off | Save plots to `plots/` directory |
| `--show-plots` | | off | Display plots interactively (requires display) |

## Architecture Overview

### Pipeline Architecture

The project uses sklearn-compatible pipelines combining three stages:
1. **Feature Extraction** - Transform raw EEG into discriminative features
2. **Preprocessing/Dimensionality Reduction** - Scaling, PCA, or CSP filtering
3. **Classification** - LDA, SVM, or Logistic Regression

Available pipelines (defined in [pipeline.py](src/pipeline.py)):
- `csp_lda` - CSP + LDA (recommended, best for motor imagery)
- `csp_svm` - CSP + SVM
- `csp_lr` - CSP + Logistic Regression
- `csp_rf` - CSP + Random Forest
- `psd_lda` - Power Spectral Density + LDA
- `bandpower_lda` - Band Power features + LDA
- `flat_pca_lda` - Flattened raw signal + PCA + LDA
- `wavelet_lda` - Wavelet (CWT Morlet) features + LDA
- `wavelet_custom` - Wavelet + PCA + MyNearestCentroid
- `csp_custom` - CSP + MyNearestCentroid

### Module Organization

The codebase is organized into domain-specific submodules:

**Root wrapper**:
- `mybci.py` - Root-level wrapper that delegates to `src/mybci.py`; enables running without `cd src`

**Core modules** (in `src/`):
- `mybci.py` - Main CLI entry point with argument parsing
- `preprocess.py` - EEG data loading and bandpass filtering (7-30Hz)
- `features.py` - Feature extractors (PSD, BandPower, Flatten, Wavelet)
- `display.py` - Centralized terminal output functions
- `constants.py` - Centralized configuration and magic numbers
- `pipeline.py` - Pipeline construction and registration

**Subpackages**:
- `transforms/` - Custom sklearn transformers
  - `csp.py` - MyCSP implementation (solves generalized eigenvalue problem)
  - `pca.py` - MyPCA implementation
- `classifiers/` - Custom sklearn classifiers
  - `nearest_centroid.py` - MyNearestCentroid implementation
- `training/` - Training workflow components
  - `core.py` - Cross-validation and holdout evaluation
  - `subject.py` - Subject-specific training logic
  - `comparison.py` - Multi-pipeline comparison
  - `persistence.py` - Model saving/loading with joblib
- `visualization/` - Plotting functions
  - `_base.py` - Shared helpers (figure sizing, save logic)
  - `cv_plots.py` - Cross-validation score plots
  - `metrics_plots.py` - Confusion matrices and classification reports
  - `comparison_plots.py` - Pipeline comparison charts
  - `csp_plots.py` - CSP spatial filter and eigenvalue inspection
  - `advanced_metrics_plots.py` - Per-class precision/recall/F1 and ROC/AUC curves
  - `learning_curve_plots.py` - Learning curves for overfitting/underfitting diagnosis

**Tests** mirror the src structure:
- `tests/transforms/` - Tests for CSP, PCA, and linalg
- `tests/features/` - Tests for feature extractors
- `tests/pipeline/` - Tests for pipeline construction
- `tests/classifiers/` - Tests for custom classifiers
- `conftest.py` - Pytest fixtures with synthetic EEG data generators

### Common Spatial Patterns (CSP)

The custom CSP implementation in [transforms/csp.py](src/transforms/csp.py) is central to the project:

1. **Mathematical approach**: Solves the generalized eigenvalue problem `Σ₁W = (Σ₁ + Σ₂)WΛ` where Σ₁, Σ₂ are class covariance matrices
2. **Spatial filtering**: Selects `n_components/2` filters from each end of the eigenvalue spectrum (most discriminative patterns)
3. **Output features**: Returns log-variance of filtered signals (optimal for LDA)
4. **sklearn compatible**: Implements `fit()` and `transform()` for pipeline integration

### Data Flow

1. **Loading** ([preprocess.py](src/preprocess.py)): Download from Physionet, extract runs 3-14
2. **Preprocessing**: Bandpass filter 7-30Hz (mu and beta bands for motor imagery)
3. **Feature Extraction**: Apply selected feature extractor (CSP, PSD, etc.)
4. **Classification**: Train classifier on extracted features
5. **Persistence**: Save trained pipeline to `models/model_s{subject}_r{run}_{pipeline}.pkl`

### Run Types and Event Mapping

The Physionet dataset has specific run types (see [constants.py](src/constants.py)):
- Runs 3,7,11: Motor execution (left fist vs right fist)
- Runs 4,8,12: Motor imagery (left fist vs right fist)
- Runs 5,9,13: Motor execution (both fists vs both feet)
- Runs 6,10,14: Motor imagery (both fists vs both feet)

Event IDs are automatically mapped in [preprocess.py](src/preprocess.py) based on run number.

### Type Hints and Constants

- All functions have comprehensive type hints using `numpy.typing.NDArray`
- Magic numbers are centralized in [constants.py](src/constants.py) (e.g., `EEG_SAMPLING_RATE=160.0`, `TARGET_ACCURACY=0.60`)
- Use `from constants import CONSTANT_NAME` rather than hardcoding values

### Testing Strategy

Tests use synthetic EEG data to avoid downloading from Physionet during CI. The fixtures in [conftest.py](tests/conftest.py) generate 2-class EEG data with configurable dimensions and spatial patterns.

Key test patterns:
- Parameterized tests for multiple pipeline configurations
- Synthetic data with known properties (e.g., separable classes for CSP)
- Round-trip serialization tests for model persistence
- Real-time prediction simulation tests

## Code Quality Standards

### Pre-commit Hooks

Pre-commit hooks run automatically on commit (configured in [.pre-commit-config.yaml](.pre-commit-config.yaml)):
- Trailing whitespace removal
- End-of-file fixer
- YAML validation
- flake8 style checking (max line length: 100)
- mypy type checking (excludes tests/)

If pre-commit fails, fix the issues and commit again.

### Style Guidelines

- **Line length**: 100 characters max (configured in [.flake8](.flake8))
- **Docstrings**: NumPy style for all public functions
- **Type hints**: Required for all function signatures
- **Imports**: Use absolute imports from `src/` (e.g., `from constants import ...`)

### mypy Configuration

mypy is configured in [setup.cfg](setup.cfg) with:
- `ignore_missing_imports = True` (for scientific libraries)
- `check_untyped_defs = True`
- Tests excluded from type checking (`[mypy-tests.*]`)

## Performance Targets

- **Minimum accuracy**: 60% on test data (TARGET_ACCURACY in constants.py)
- **Prediction time**: < 2 seconds per epoch (MAX_PREDICTION_TIME)

## Development Workflow

1. **Environment setup**: `uv sync`
2. **Install pre-commit**: `uv run pre-commit install`
3. **Make changes**: Follow PEP 8, add type hints, write NumPy-style docstrings
4. **Add tests**: Use synthetic data from conftest.py fixtures
5. **Run quality checks**: `uv run pre-commit run --all-files && uv run pytest tests/ -v`
6. **Commit**: Pre-commit hooks run automatically

## Visualization Outputs

When plots are enabled, the following files are generated in `plots/`:

| File | Plot type | Generated by |
|------|-----------|--------------|
| `cv_scores_s{S}_r{R}_{P}.png` | Cross-validation fold scores | `cv_plots.py` |
| `confusion_matrix_s{S}_r{R}_{P}.png` | Confusion matrix | `metrics_plots.py` |
| `classification_report_s{S}_r{R}_{P}.png` | Per-class precision/recall table | `metrics_plots.py` |
| `pipeline_comparison_s{S}_r{R}.png` | All pipelines accuracy bar chart | `comparison_plots.py` |
| `csp_filters_s{S}_r{R}_{P}.png` | CSP spatial filter topomaps | `csp_plots.py` |
| `advanced_metrics_s{S}_r{R}_{P}.png` | ROC curve + per-class F1 | `advanced_metrics_plots.py` |
| `learning_curve_s{S}_r{R}_{P}.png` | Train/val score vs. training size | `learning_curve_plots.py` |

Note: `--show-plots` opens plots interactively and requires a graphical display (not available in headless/CI environments).

## Important Notes

- The root `mybci.py` wrapper allows running from the project root without `cd src`
- Model files are saved to `models/` directory with naming convention `model_s{subject}_r{run}_{pipeline}.pkl`
- Plots are saved to `plots/` directory when using `--save-plots`
- Use `uv sync` with `uv.lock` for reproducible builds
