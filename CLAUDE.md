# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Total Perspective Vortex is an EEG Brain-Computer Interface (BCI) system for motor imagery classification using the Physionet EEGMMIDB dataset. It implements a complete BCI pipeline from preprocessing to real-time prediction.

## Development Commands

### Testing
```bash
# Run all tests (148 tests total)
pytest tests/ -v

# Run specific test file
pytest tests/test_mycsp.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Run flake8 style checking
flake8 src/ tests/

# Run mypy type checking
mypy src/ --ignore-missing-imports

# Run all pre-commit hooks manually
pre-commit run --all-files
```

### Training and Prediction
```bash
# Train a model (from src/ directory)
cd src
python mybci.py <subject> <run> train [--pipeline csp_lda] [--cv 5]

# Run prediction
python mybci.py <subject> <run> predict

# Compare all pipelines
python mybci.py 4 14 train --compare

# Generate demo plots
cd ..  # back to root
python demo_plots.py 4 14
```

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
- `psd_lda` - Power Spectral Density + LDA
- `bandpower_lda` - Band Power features + LDA
- `flat_pca_lda` - Flattened raw signal + PCA + LDA

### Module Organization

The codebase is organized into domain-specific submodules:

**Core modules** (in `src/`):
- `mybci.py` - Main CLI entry point with argument parsing
- `preprocess.py` - EEG data loading and bandpass filtering (7-30Hz)
- `features.py` - Feature extractors (PSD, BandPower, Flatten)
- `constants.py` - Centralized configuration and magic numbers
- `pipeline.py` - Pipeline construction and registration

**Subpackages**:
- `transforms/` - Custom sklearn transformers
  - `csp.py` - MyCSP implementation (solves generalized eigenvalue problem)
  - `pca.py` - MyPCA implementation
- `training/` - Training workflow components
  - `core.py` - Cross-validation and holdout evaluation
  - `subject.py` - Subject-specific training logic
  - `comparison.py` - Multi-pipeline comparison
  - `persistence.py` - Model saving/loading with joblib
- `visualization/` - Plotting functions
  - `cv_plots.py` - Cross-validation score plots
  - `metrics_plots.py` - Confusion matrices and classification reports
  - `comparison_plots.py` - Pipeline comparison charts

**Tests** mirror the src structure:
- `tests/transforms/` - Tests for CSP and PCA
- `tests/features/` - Tests for feature extractors
- `tests/pipeline/` - Tests for pipeline construction
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

1. **Environment setup**: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements-lock.txt`
2. **Install pre-commit**: `pre-commit install`
3. **Make changes**: Follow PEP 8, add type hints, write NumPy-style docstrings
4. **Add tests**: Use synthetic data from conftest.py fixtures
5. **Run quality checks**: `pre-commit run --all-files && pytest tests/ -v`
6. **Commit**: Pre-commit hooks run automatically

## Important Notes

- The CLI must be run from the `src/` directory due to path manipulation in mybci.py
- Model files are saved to `models/` directory with naming convention `model_s{subject}_r{run}_{pipeline}.pkl`
- Plots are saved to `plots/` directory when using `--save-plots`
- Use `requirements-lock.txt` for reproducible builds (includes exact versions)
