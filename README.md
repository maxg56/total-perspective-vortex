# Total Perspective Vortex

EEG Brain-Computer Interface (BCI) system for motor imagery classification using the Physionet EEGMMIDB dataset.

## Overview

This project implements a complete BCI pipeline:
1. **Preprocessing**: Load and filter EEG data from Physionet
2. **Feature Extraction**: PSD, band power, or CSP features
3. **Classification**: LDA, SVM, or Logistic Regression
4. **Real-time Prediction**: Simulate streaming EEG classification

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
├── mybci.py        # Main CLI entry point
├── preprocess.py   # Data loading and filtering
├── features.py     # Feature extraction (PSD, band power)
├── mycsp.py        # Custom CSP implementation
├── pipeline.py     # sklearn pipeline construction
├── train.py        # Training and cross-validation
└── predict.py      # Real-time prediction simulation

models/             # Saved models directory
```

## Custom CSP Implementation

The `mycsp.py` module contains a custom implementation of Common Spatial Patterns (CSP) as an sklearn transformer:

- Computes class-specific covariance matrices
- Solves generalized eigenvalue problem
- Selects most discriminative spatial filters
- Returns log-variance features for classification

## Target Performance

- **Minimum accuracy**: 60% on test data
- **Prediction time**: < 2 seconds per epoch

## License

MIT License
