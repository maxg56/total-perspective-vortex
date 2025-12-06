# Visualization Guide

This document explains how to use the visualization features in the Total Perspective Vortex EEG BCI system.

## Quick Start

### Using the CLI

Train a model with visualization enabled:

```bash
cd src
python mybci.py 4 14 train --save-plots
```

This will:
- Train the model
- Generate plots automatically
- Save them to the `plots/` directory

### Using the Demo Script

Run the demonstration script:

```bash
python demo_plots.py 4 14
```

This provides a comprehensive visualization demo with:
- Cross-validation scores
- Confusion matrix
- Training summary with all metrics

## Available Visualizations

### 1. Cross-Validation Scores

**File**: `plots/cv_scores_<pipeline>.png`

Shows the accuracy for each cross-validation fold with:
- Bar chart for each fold's accuracy
- Mean accuracy line (red dashed)
- Target accuracy line (green dashed at 60%)
- Individual values displayed on bars

**When to use**: To understand model consistency across different data splits.

### 2. Confusion Matrix

**File**: `plots/confusion_matrix_<pipeline>.png`

Heatmap showing prediction results with:
- True vs predicted labels
- Color intensity indicating frequency
- Exact counts in each cell
- Useful for identifying which classes are confused

**When to use**: To diagnose classification errors and class-specific performance.

### 3. Training Summary

**File**: `plots/training_summary_<pipeline>.png`

Comprehensive 3-panel visualization showing:
- **Panel 1**: Cross-validation scores (bar chart)
- **Panel 2**: Confusion matrix (heatmap)
- **Panel 3**: Per-class accuracy (bar chart)

**When to use**: For a complete overview of model performance.

### 4. Pipeline Comparison

**File**: `plots/pipeline_comparison.png`

Bar chart comparing multiple pipelines with:
- Mean accuracy for each pipeline
- Error bars (standard deviation)
- Best pipeline highlighted in gold
- Target line at 60%

**When to use**: When deciding which pipeline to use (with `--compare` option).

### 5. Detailed CV Comparison

**File**: `plots/pipeline_comparison_detailed.png`

Box plots showing distribution of CV scores for each pipeline:
- Median, quartiles, and outliers
- Mean line
- Target accuracy line

**When to use**: For detailed comparison of pipeline stability.

## Programmatic Usage

### Basic Training with Plots

```python
from preprocess import preprocess_subject
from train import train_and_evaluate

# Load data
X, y, epochs = preprocess_subject(subject=4, runs=[14])

# Train with plots (displayed but not saved)
pipeline, scores = train_and_evaluate(
    X, y,
    pipeline_name='csp_lda',
    plot=True,
    save_plots=False
)
```

### Training with Holdout and Saved Plots

```python
from preprocess import preprocess_subject
from train import train_with_holdout

# Load data
X, y, epochs = preprocess_subject(subject=4, runs=[14])

# Train with holdout validation and save all plots
pipeline, cv_scores, test_acc = train_with_holdout(
    X, y,
    pipeline_name='csp_lda',
    test_size=0.2,
    cv=5,
    plot=True,
    save_plots=True
)
```

### Comparing Pipelines

```python
from preprocess import preprocess_subject
from train import compare_pipelines

# Load data
X, y, epochs = preprocess_subject(subject=4, runs=[14])

# Compare all pipelines with plots
results = compare_pipelines(
    X, y,
    cv=5,
    plot=True,
    save_plots=True
)
```

### Custom Visualizations

You can also use the visualization functions directly:

```python
from visualization import (
    plot_cv_scores,
    plot_confusion_matrix,
    plot_training_summary
)
import numpy as np

# Example: Plot custom CV scores
scores = np.array([0.8, 0.85, 0.9, 0.82, 0.88])
plot_cv_scores(
    scores,
    title="My Custom CV Results",
    save_path="my_custom_plot.png",
    show=False
)

# Example: Plot confusion matrix
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 2, 1, 2, 1, 2])
y_pred = np.array([1, 2, 1, 1, 1, 2])

plot_confusion_matrix(
    y_true, y_pred,
    class_names=['Hands', 'Feet'],
    save_path="my_confusion_matrix.png"
)
```

## Command-Line Options

### Training with Visualization

```bash
# Save plots (default: plots not saved)
python mybci.py 4 14 train --save-plots

# Disable plotting completely
python mybci.py 4 14 train --no-plot

# Compare pipelines with plots
python mybci.py 4 14 train --compare --save-plots
```

## Output Directory

All plots are saved to the `plots/` directory relative to where you run the script:

```
plots/
├── cv_scores_csp_lda.png
├── cv_scores_csp_lda_holdout.png
├── confusion_matrix_csp_lda.png
├── training_summary_csp_lda.png
├── pipeline_comparison.png
└── pipeline_comparison_detailed.png
```

## Plot Customization

### Change Plot Backend

By default, plots use the `Agg` backend (non-interactive). To change:

```python
# In your script, before importing visualization
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', etc.

from visualization import plot_cv_scores
```

### Display Plots Interactively

```python
# Set show=True to display plots
plot_cv_scores(scores, show=True)
```

## Tips

1. **Performance**: Plot generation is fast and doesn't significantly slow down training.

2. **Disk Space**: Plots are saved as PNG files (~80-170 KB each).

3. **Reproducibility**: Always save plots when running experiments for later reference.

4. **Comparison**: Use `--compare` to automatically generate comparison plots.

5. **Debugging**: If plots aren't appearing, check that matplotlib is properly installed.

## Troubleshooting

### Plots not saved

- Check that you're using `--save-plots` flag or `save_plots=True`
- Verify write permissions in the current directory
- Check that the `plots/` directory was created

### Import errors

```bash
# Make sure matplotlib is installed
pip install matplotlib
```

### Backend errors

If you get backend-related errors:

```python
# Set backend before importing
import matplotlib
matplotlib.use('Agg')
```

## Examples

See `demo_plots.py` for a complete working example showing all visualization features.
