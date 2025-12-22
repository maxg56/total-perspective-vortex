# Total Perspective Vortex

EEG Brain-Computer Interface (BCI) system for motor imagery classification using the Physionet EEGMMIDB dataset.

## Overview

This project implements a complete BCI pipeline:
1. **Preprocessing**: Load and filter EEG data from Physionet
2. **Feature Extraction**: PSD, band power, or CSP features
3. **Classification**: LDA, SVM, or Logistic Regression
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

CSP features are computed as the **log-variance of spatially filtered signals** ([transforms/csp.py:187-189](src/transforms/csp.py)). The logarithm transforms the multiplicative variance structure into an additive one, making the feature distribution approximately Gaussian—a critical assumption for optimal Linear Discriminant Analysis performance.

**Oral explanation**: *Log-variance transformation normalizes CSP feature distributions to be approximately Gaussian. This mathematical property directly aligns with LDA's assumption of normally distributed features, maximizing classifier performance.*

### LDA Classifier

Linear Discriminant Analysis ([pipeline.py](src/pipeline.py)) is the standard classifier for CSP-based BCIs because it assumes **Gaussian class-conditional distributions**—precisely what log-variance CSP features provide. LDA is computationally efficient, requires no hyperparameter tuning, and finds the optimal linear decision boundary when its Gaussian assumption holds, making it ideal for real-time BCI applications.

**Oral explanation**: *LDA is optimal for Gaussian-distributed features like log-variance CSP outputs. It requires minimal computation and no hyperparameter tuning, making it suitable for real-time brain-computer interfaces.*

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model on a specific subject and run:

```bash
cd src
python mybci.py <subject> <run> train
```

Example:
```bash
python mybci.py 4 14 train
```

### Prediction

Run prediction with a trained model:

```bash
python mybci.py <subject> <run> predict
```

Example:
```bash
python mybci.py 4 14 predict
```

### Options

```
--pipeline, -p     Pipeline to use (csp_lda, csp_svm, csp_lr, psd_lda, bandpower_lda, flat_pca_lda)
--cv               Number of cross-validation folds (default: 5)
--compare          Compare all pipelines and use the best
--n-components     Number of CSP components (default: 6)
--model-dir        Directory for saved models (default: models)
```

### Visualization

#### Raw Signal Visualization

Visualize raw and filtered EEG signals with power spectral density analysis:

```bash
# Display raw and filtered EEG signals with PSD
python demo_raw_signal.py [subject] [run]

# Example: Subject 4, Run 14
python demo_raw_signal.py 4 14
```

This demonstration script shows:
- **Raw EEG signal traces**: Unfiltered signal from all channels
- **Filtered signal traces**: 7-30 Hz bandpass filtered signal (mu and beta rhythms)
- **Power Spectral Density (PSD)**: Comparison of raw vs filtered signals in the 7-30 Hz range
- **Topographic PSD maps**: Spatial distribution of power across frequency bands

This is particularly useful for:
- Understanding the preprocessing pipeline
- Validating bandpass filtering (7-30 Hz for motor imagery)
- Demonstrating signal quality for presentations and defenses
- Analyzing frequency band characteristics (mu: 8-12 Hz, beta: 12-30 Hz)

#### Training Performance Visualization

The training process automatically generates visualizations to track performance:

```bash
# Generate plots during training (saved to plots/ directory)
python demo_plots.py 4 14
```

Available visualizations:
- **Cross-validation scores**: Bar chart showing accuracy for each CV fold
- **Confusion matrix**: Heatmap of prediction results on test set
- **Training summary**: Comprehensive 3-panel view with CV scores, confusion matrix, and per-class accuracy
- **Pipeline comparison**: Compare performance of different models (when using --compare)

Programmatic usage:

```python
from src.train import train_with_holdout
from src.preprocess import preprocess_subject

# Load data
X, y, epochs = preprocess_subject(subject=4, runs=[14])

# Train with visualizations
pipeline, cv_scores, test_acc = train_with_holdout(
    X, y,
    pipeline_name='csp_lda',
    plot=True,        # Enable plotting
    save_plots=True   # Save plots to disk
)
```

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
├── mybci.py          # Main CLI entry point
├── preprocess.py     # Data loading and filtering
├── features.py       # Feature extraction (PSD, band power)
├── mycsp.py          # Custom CSP implementation
├── pipeline.py       # sklearn pipeline construction
├── train.py          # Training and cross-validation
├── predict.py        # Real-time prediction simulation
└── visualization.py  # Plotting functions for results

tests/
├── conftest.py         # Pytest fixtures and synthetic data generators
├── test_preprocess.py  # Tests for EEG preprocessing
├── test_features.py    # Tests for feature extractors
├── test_mycsp.py       # Tests for CSP and PCA implementations
├── test_pipeline.py    # Tests for pipeline construction
├── test_train.py       # Tests for training and model persistence
└── test_predict.py     # Tests for prediction functions

models/             # Saved models directory
plots/              # Generated visualization plots
demo_plots.py       # Demonstration script for training visualizations
demo_raw_signal.py  # Demonstration script for raw signal visualization
```

## Testing

The project includes a comprehensive test suite with 148 tests using pytest.

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mycsp.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| `test_preprocess.py` | 11 | Run type identification, event IDs, parameter validation |
| `test_features.py` | 29 | PSD, BandPower, LogVariance, Flatten extractors |
| `test_mycsp.py` | 31 | CSP fitting, transform, covariance computation, PCA |
| `test_pipeline.py` | 26 | All 6 pipeline configurations, sklearn compatibility |
| `test_train.py` | 24 | Cross-validation, holdout training, model save/load |
| `test_predict.py` | 27 | Single/batch prediction, real-time simulation |

### Synthetic Data

Tests use synthetic EEG data to avoid downloading from Physionet during testing. The fixtures in `conftest.py` generate:
- 2-class EEG data with different spatial patterns
- Configurable dimensions (epochs, channels, time points)
- Pre-trained pipelines for prediction tests

## Custom CSP Implementation

The `mycsp.py` module contains a custom implementation of Common Spatial Patterns (CSP) as an sklearn transformer:

- Computes class-specific covariance matrices
- Solves generalized eigenvalue problem
- Selects most discriminative spatial filters
- Returns log-variance features for classification

## Target Performance

- **Minimum accuracy**: 60% on test data
- **Prediction time**: < 2 seconds per epoch

## Experimental Validation

To prove that the system achieves the ≥60% accuracy threshold across multiple subjects (required for Issue #14), use the provided automated experimental validation script:

### Quick Start

```bash
# Run all required experiments automatically
bash run_experiments.sh
```

This script will:
1. Train models for subjects 1, 4, and 10 with 5-fold cross-validation
2. Evaluate on holdout test sets
3. Measure prediction times
4. Generate comprehensive results and plots
5. Validate that the ≥60% accuracy target is met

### Manual Validation

If you prefer to run experiments individually:

```bash
# Subject 1, Run 6 (hands vs feet motor imagery)
cd src
python mybci.py 1 6 train --cv 5
python mybci.py 1 6 predict

# Subject 4, Run 14 (hands vs feet motor imagery)
python mybci.py 4 14 train --cv 5
python mybci.py 4 14 predict

# Subject 10, Run 6 (hands vs feet motor imagery)
python mybci.py 10 6 train --cv 5
python mybci.py 10 6 predict
```

### Results Documentation

After running experiments, results are documented in:
- **EXPERIMENTAL_RESULTS.md**: Complete experimental results and analysis
- **results/**: Log files with detailed metrics
- **plots/**: Visualizations (confusion matrices, CV scores, etc.)
- **models/**: Trained model files

See [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md) for detailed information about:
- Experimental protocol
- Per-subject results (CV accuracy, test accuracy, prediction times)
- Aggregate statistics
- Target achievement validation
- Troubleshooting network/environment issues

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd total-perspective-vortex

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (including development tools)
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### Code Quality Tools

This project uses several tools to maintain code quality:

- **flake8**: Style guide enforcement (PEP 8)
- **mypy**: Static type checking
- **pre-commit**: Automatic checks before commits

### Running Code Quality Checks

```bash
# Run flake8
flake8 src/ tests/

# Run mypy for type checking
mypy src/ --ignore-missing-imports

# Run all pre-commit hooks manually
pre-commit run --all-files

# Run tests
pytest tests/ -v
```

### Pre-commit Hooks

Pre-commit hooks automatically run on every commit. They will:
- Check for trailing whitespace
- Ensure files end with newline
- Validate YAML files
- Run flake8 for style checking
- Run mypy for type checking

If any check fails, the commit will be rejected. Fix the issues and commit again.

### Type Hints

All modules include comprehensive type hints for better code documentation and static analysis:

```python
from typing import List, Tuple, Optional
from numpy.typing import NDArray
import numpy as np

def preprocess_subject(
    subject: int,
    runs: List[int]
) -> Tuple[NDArray[np.float64], NDArray[np.int64], mne.Epochs]:
    ...
```

### Configuration Files

- `.flake8`: Flake8 configuration (line length, ignored errors)
- `setup.cfg`: Mypy configuration
- `.pre-commit-config.yaml`: Pre-commit hooks configuration

### Contributing Guidelines

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes**
   - Follow PEP 8 style guide
   - Add type hints to new functions
   - Write docstrings (NumPy style)
   - Add tests for new functionality

3. **Run quality checks**
   ```bash
   pre-commit run --all-files
   pytest tests/ -v
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```
   Pre-commit hooks will run automatically.

5. **Push and create Pull Request**
   ```bash
   git push origin feature/my-feature
   ```

## License

MIT License
