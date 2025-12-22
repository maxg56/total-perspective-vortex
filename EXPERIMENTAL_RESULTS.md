# Experimental Results - Total Perspective Vortex BCI System

**Issue Reference:** #14 - Missing Experimental Results (CRITICAL)
**Objective:** Prove that the system achieves ≥60% accuracy across multiple subjects
**Date Created:** 2025-12-22
**Status:** ⚠️ **EXPERIMENTS NEED TO BE RUN**

---

## Executive Summary

This document contains the experimental results required to demonstrate that the Total Perspective Vortex BCI system achieves the target accuracy threshold of ≥60% across multiple subjects and runs.

**Target Performance:** ≥ 60% accuracy (as defined in `constants.py:TARGET_ACCURACY`)

---

## Experimental Protocol

### Subjects and Runs Tested

Following issue #14 requirements, experiments must be conducted on:

1. **Subject 1, Run 6** - Motor imagery (both fists vs both feet)
2. **Subject 4, Run 14** - Motor imagery (both fists vs both feet)
3. **Subject 10, Run 6** - Motor imagery (both fists vs both feet)

### Pipeline Configuration

- **Feature Extraction:** Common Spatial Patterns (CSP)
- **Classifier:** Linear Discriminant Analysis (LDA)
- **Pipeline Name:** `csp_lda`
- **Cross-Validation:** 5-fold stratified CV
- **Test Set:** 20% holdout for final evaluation

### Preprocessing

- **Bandpass Filter:** 7-30 Hz (mu and beta bands)
- **Epoching:** Event-based segmentation
- **Channel Selection:** All 64 EEG channels
- **Sampling Rate:** 160 Hz

---

## How to Run the Experiments

### Quick Start

Use the provided script to run all required experiments:

```bash
# From the project root directory
bash run_experiments.sh
```

This will automatically:
1. Train models for all 3 subjects/runs with 5-fold CV
2. Evaluate on holdout test sets
3. Measure prediction times
4. Generate this results file with actual data
5. Save all plots to `plots/`

### Manual Execution

If you prefer to run experiments individually:

```bash
# Subject 1, Run 6
cd src
python mybci.py 1 6 train --cv 5
python mybci.py 1 6 predict

# Subject 4, Run 14
python mybci.py 4 14 train --cv 5
python mybci.py 4 14 predict

# Subject 10, Run 6
python mybci.py 10 6 train --cv 5
python mybci.py 10 6 predict
```

---

## Results

### Subject 1, Run 6 (Hands vs Feet Motor Imagery)

**Model:** `model_s1_r6_csp_lda.pkl`

| Metric | Value |
|--------|-------|
| Cross-Validation Mean Accuracy | **TODO: RUN EXPERIMENT** |
| Cross-Validation Std Dev | **TODO: RUN EXPERIMENT** |
| Test Set Accuracy | **TODO: RUN EXPERIMENT** |
| Training Time | **TODO: RUN EXPERIMENT** |
| Average Prediction Time (per epoch) | **TODO: RUN EXPERIMENT** |
| Maximum Prediction Time (per epoch) | **TODO: RUN EXPERIMENT** |

**Cross-Validation Scores:**
```
Fold 1: TODO
Fold 2: TODO
Fold 3: TODO
Fold 4: TODO
Fold 5: TODO
```

**Status:** ⚠️ Experiment not yet run
**Meets Target (≥60%):** N/A

---

### Subject 4, Run 14 (Hands vs Feet Motor Imagery)

**Model:** `model_s4_r14_csp_lda.pkl`

| Metric | Value |
|--------|-------|
| Cross-Validation Mean Accuracy | **TODO: RUN EXPERIMENT** |
| Cross-Validation Std Dev | **TODO: RUN EXPERIMENT** |
| Test Set Accuracy | **TODO: RUN EXPERIMENT** |
| Training Time | **TODO: RUN EXPERIMENT** |
| Average Prediction Time (per epoch) | **TODO: RUN EXPERIMENT** |
| Maximum Prediction Time (per epoch) | **TODO: RUN EXPERIMENT** |

**Cross-Validation Scores:**
```
Fold 1: TODO
Fold 2: TODO
Fold 3: TODO
Fold 4: TODO
Fold 5: TODO
```

**Status:** ⚠️ Experiment not yet run
**Meets Target (≥60%):** N/A

---

### Subject 10, Run 6 (Hands vs Feet Motor Imagery)

**Model:** `model_s10_r6_csp_lda.pkl`

| Metric | Value |
|--------|-------|
| Cross-Validation Mean Accuracy | **TODO: RUN EXPERIMENT** |
| Cross-Validation Std Dev | **TODO: RUN EXPERIMENT** |
| Test Set Accuracy | **TODO: RUN EXPERIMENT** |
| Training Time | **TODO: RUN EXPERIMENT** |
| Average Prediction Time (per epoch) | **TODO: RUN EXPERIMENT** |
| Maximum Prediction Time (per epoch) | **TODO: RUN EXPERIMENT** |

**Cross-Validation Scores:**
```
Fold 1: TODO
Fold 2: TODO
Fold 3: TODO
Fold 4: TODO
Fold 5: TODO
```

**Status:** ⚠️ Experiment not yet run
**Meets Target (≥60%):** N/A

---

## Aggregate Results

### Summary Statistics

| Metric | Value |
|--------|-------|
| **Overall Mean CV Accuracy** | **TODO: Calculate after experiments** |
| **Overall Mean Test Accuracy** | **TODO: Calculate after experiments** |
| **Subjects Meeting Target (≥60%)** | **TODO: Count after experiments** |
| **Success Rate** | **TODO: Calculate percentage** |

### Performance Target Achievement

✅ / ❌ **Target Achieved:** The system achieves ≥60% accuracy across multiple subjects.

_(This will be automatically filled after running experiments)_

---

## Generated Visualizations

The following plots are generated during the experimental run:

- `plots/subject1_run6_cv_scores.png` - Cross-validation fold scores
- `plots/subject1_run6_confusion_matrix.png` - Test set confusion matrix
- `plots/subject1_run6_training_summary.png` - Combined metrics summary

- `plots/subject4_run14_cv_scores.png`
- `plots/subject4_run14_confusion_matrix.png`
- `plots/subject4_run14_training_summary.png`

- `plots/subject10_run6_cv_scores.png`
- `plots/subject10_run6_confusion_matrix.png`
- `plots/subject10_run6_training_summary.png`

- `plots/aggregate_results.png` - Comparison across all subjects

---

## Statistical Analysis

### Confidence Intervals

95% confidence intervals for mean accuracy:

- **Subject 1, Run 6:** TODO ± TODO
- **Subject 4, Run 14:** TODO ± TODO
- **Subject 10, Run 6:** TODO ± TODO

### Significance Testing

Comparison against chance-level performance (50% for 2-class problem):

- All subjects show statistically significant improvement over chance (p < 0.05) **[TO BE VERIFIED]**

---

## Reproducibility

### Environment

- **Python Version:** 3.11+
- **Key Dependencies:** numpy 1.26.4, scipy 1.12.0, scikit-learn 1.5.0, mne 1.6.1
- **Random Seed:** Set in `constants.py` for reproducibility
- **Dataset:** Physionet EEG Motor Movement/Imagery Dataset (EEGMMIDB)

### Data Availability

- **Source:** https://physionet.org/content/eegmmidb/1.0.0/
- **Automatic Download:** MNE library handles data fetching
- **Cache Location:** `~/mne_data/MNE-eegbci-data/`

---

## Troubleshooting

### Network Issues

If data download fails due to proxy/network restrictions:

1. Check proxy settings: `env | grep -i proxy`
2. Try unsetting proxy: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`
3. Verify DNS resolution: `nslookup physionet.org`
4. Manually download data from PhysioNet website if needed

### Out of Memory

If experiments fail due to memory constraints:

- Reduce cross-validation folds: `--cv 3` instead of `--cv 5`
- Process one subject at a time
- Close other applications

### Slow Performance

- Ensure NumPy is using optimized BLAS/LAPACK libraries
- Check CPU usage during training
- Consider using a machine with more CPU cores

---

## Conclusion

**Final Verdict:** ⚠️ **PENDING - EXPERIMENTS MUST BE RUN**

Once experiments are completed:

- ✅ All required subjects (1, 4, 10) will be tested
- ✅ Results will demonstrate ≥60% accuracy threshold achievement
- ✅ Comprehensive documentation will be available for presentation/defense
- ✅ Generated plots will provide visual evidence of performance

**Next Steps:**

1. Run `bash run_experiments.sh` to execute all experiments
2. Review generated results in this file
3. Examine plots in `plots/` directory
4. Include this document and plots in project defense/presentation

---

## References

1. Goldberger, A., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." Circulation.
2. Schalk, G., et al. (2004). "BCI2000: A general-purpose brain-computer interface (BCI) system." IEEE Transactions on Biomedical Engineering.
3. Blankertz, B., et al. (2008). "Optimizing spatial filters for robust EEG single-trial analysis." IEEE Signal Processing Magazine.

---

**Document Version:** 1.0
**Last Updated:** 2025-12-22
**Auto-Generated:** No (Template - awaiting experimental data)
**Requires Action:** YES - Run experiments to populate results
