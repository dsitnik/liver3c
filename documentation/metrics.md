# Metrics Calculation and Analysis Guide

This document provides comprehensive instructions for computing, analyzing, and interpreting evaluation metrics for liver cancer segmentation models.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Metrics Computed](#metrics-computed)
- [Usage](#usage)
- [Output Files](#output-files)
- [Metric Interpretation](#metric-interpretation)
- [Advanced Analysis](#advanced-analysis)
- [Troubleshooting](#troubleshooting)

## Overview

The project provides two tools for metrics computation and analysis:

1. **`scripts/metrics/compute_test_metrics.py`**: Computes metrics for individual folds
2. **`scripts/metrics/generate_metrics_summary.py`**: Aggregates metrics across multiple folds

Both tools provide comprehensive evaluation metrics including Dice score, Jaccard index, precision, recall, and F1 score for multi-class segmentation.

## Quick Start

### Compute Metrics for a Single Fold

```bash
python scripts/metrics/compute_test_metrics.py \
    --predictions predictions/Dataset104_test/fold_0 \
    --labels nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs \
    --output predictions/Dataset104_test/fold_0/metrics.txt
```

### Generate Summary Across All Folds

```bash
python scripts/metrics/generate_metrics_summary.py \
    --results_dir predictions/Dataset104_test \
    --output summary.csv \
    --include_ensemble
```

## Metrics Computed

### Overall Metrics

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **Accuracy** | Pixel-wise classification accuracy | [0, 1] | Higher is better; proportion of correctly classified pixels |
| **Balanced Accuracy** | Average of per-class recall | [0, 1] | Better for imbalanced datasets; accounts for class imbalance |
| **Cohen's Kappa** | Agreement accounting for chance | [-1, 1] | >0.8 excellent, 0.6-0.8 substantial, 0.4-0.6 moderate |
| **Matthews Correlation Coefficient** | Balanced measure for binary/multi-class | [-1, 1] | 1=perfect, 0=random, -1=inverse |

### Per-Class Metrics

Computed for each class (Background, Metastatic, HCC, CHO):

| Metric | Formula | Description |
|--------|---------|-------------|
| **Dice Score** | `2*TP / (2*TP + FP + FN)` | Overlap between prediction and ground truth |
| **Jaccard Index (IoU)** | `TP / (TP + FP + FN)` | Intersection over Union |
| **Precision** | `TP / (TP + FP)` | Of predicted class pixels, how many are correct |
| **Recall (Sensitivity)** | `TP / (TP + FN)` | Of true class pixels, how many are detected |
| **F1 Score** | `2 * (Precision * Recall) / (Precision + Recall)` | Harmonic mean of precision and recall |
| **Support** | Count of true pixels | Number of pixels in ground truth |

### Foreground Metrics

Special metrics focusing only on tumor classes (excluding background):

| Metric | Description |
|--------|-------------|
| **Foreground Macro Dice** | Average Dice across tumor classes |
| **Foreground Macro Precision** | Average precision across tumor classes |
| **Foreground Macro Recall** | Average recall across tumor classes |
| **Foreground Micro Precision** | Precision computed on all tumor pixels together |
| **Foreground Micro Recall** | Recall computed on all tumor pixels together |

**Note**: Foreground metrics filter to only pixels where the true label is a tumor (>0), then compute metrics. This measures: "Given a pixel IS a tumor, did we classify the tumor TYPE correctly?"

## Usage

### Individual Fold Metrics

#### Basic Usage

```bash
python scripts/metrics/compute_test_metrics.py \
    --predictions <prediction_dir> \
    --labels <ground_truth_dir> \
    --output <output_file>
```

#### Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--predictions` | Yes | Directory with predicted masks | - |
| `--labels` | Yes | Directory with ground truth masks | - |
| `--output` | No | Output file path | `test_results.txt` |
| `--num_classes` | No | Number of segmentation classes | `4` |
| `--checkpoint_type` | No | Checkpoint type for report | `best` |

#### Example with All Options

```bash
python scripts/metrics/compute_test_metrics.py \
    --predictions predictions/Dataset104_test/fold_0 \
    --labels nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs \
    --output predictions/Dataset104_test/fold_0/evaluation_report.txt \
    --num_classes 4 \
    --checkpoint_type best
```

### Multi-Fold Summary

#### Basic Usage

```bash
python scripts/metrics/generate_metrics_summary.py \
    --results_dir <directory_with_folds> \
    --output <output_csv>
```

#### Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--results_dir` | Yes | Directory containing fold_X subdirectories | - |
| `--output` | No | Output CSV file path | `metrics_summary.csv` |
| `--include_ensemble` | No | Include ensemble metrics if available | False |

#### Example: Complete Summary

```bash
python scripts/metrics/generate_metrics_summary.py \
    --results_dir predictions/Dataset104_test \
    --output complete_summary.csv \
    --include_ensemble
```

## Output Files

### Per-Fold Outputs (from `scripts/metrics/compute_test_metrics.py`)

#### 1. Text Report (`test_evaluation_report.txt`)

Human-readable report with:
- Overall metrics (accuracy, kappa, MCC)
- Foreground metrics (macro/micro precision, recall, F1)
- Per-class metrics (Dice, Jaccard, precision, recall, F1)
- Confusion matrix
- Per-case statistics (mean ± std)

Example:
```
================================================================================
TEST SET EVALUATION REPORT
================================================================================
Number of test cases: 36

OVERALL METRICS:
----------------------------------------
Accuracy: 0.9143
Balanced Accuracy: 0.8683
Cohen's Kappa: 0.8244
Matthews Correlation Coefficient: 0.8251

FOREGROUND-ONLY METRICS (Tumor Classes):
----------------------------------------
Macro Dice: 0.8325
Macro Precision: 0.9832
Macro Recall: 0.8476
...
```

#### 2. JSON Report (`test_evaluation_report.json`)

Machine-readable format containing:
- All aggregate metrics
- Per-case summary statistics
- Confusion matrix

#### 3. Per-Case CSV (`test_evaluation_report_per_case.csv`)

Spreadsheet with one row per test image:
- `case_id`: Image filename
- `accuracy`: Overall pixel accuracy
- `Background_Dice`, `Metastatic_Dice`, `HCC_Dice`, `CHO_Dice`: Per-class Dice scores
- `foreground_accuracy`: Accuracy on tumor pixels only

Example:
```csv
case_id,accuracy,foreground_accuracy,Background_Dice,Metastatic_Dice,HCC_Dice,CHO_Dice
cho_14a,0.9523,0.8234,0.9612,0.0,0.0,0.8234
metastatic_5,0.9234,0.9456,0.9445,0.9456,0.0,0.0
...
```

#### 4. Aggregate CSV (`test_evaluation_report_aggregate.csv`)

Summary of all aggregate metrics in structured format.

### Multi-Fold Outputs (from `scripts/metrics/generate_metrics_summary.py`)

#### 1. Summary CSV (`metrics_summary.csv`)

Contains:
- One row per fold (fold_0 through fold_9)
- **Mean row**: Mean of each metric across all folds
- **Std Dev row**: Standard deviation of each metric
- **Ensemble row** (if `--include_ensemble`): Ensemble performance

Example structure:
```csv
Fold,Accuracy,Foreground_Macro_Dice,Metastatic_Dice,HCC_Dice,CHO_Dice,...
fold_0,0.9143,0.8325,0.8995,0.8209,0.7771,...
fold_1,0.9132,0.8330,0.9021,0.8134,0.7835,...
...
fold_9,0.9118,0.8305,0.9009,0.8105,0.7801,...
Mean,0.9155,0.8379,0.9013,0.8249,0.7875,...
Std Dev,0.0038,0.0104,0.0027,0.0165,0.0150,...
Ensemble,0.9234,0.8567,0.9123,0.8445,0.8234,...
```

#### 2. Transposed Summary (`metrics_summary_transposed.csv`)

Same data but transposed (metrics as rows, folds as columns) for easier reading.

## Metric Interpretation

### Dice Score

**Primary metric for segmentation tasks**

- **0.0**: No overlap
- **0.5**: Moderate overlap
- **0.7**: Good overlap
- **0.85+**: Excellent overlap
- **1.0**: Perfect match

**Clinical interpretation:**
- Metastatic: >0.85 excellent, 0.75-0.85 good
- HCC: >0.80 excellent, 0.70-0.80 good
- CHO: >0.75 excellent, 0.65-0.75 good

### Precision vs Recall Trade-off

- **High Precision, Low Recall**: Model is conservative (misses lesions but rarely gives false positives)
- **Low Precision, High Recall**: Model is aggressive (detects most lesions but many false positives)
- **Balanced**: Both ~0.85+

**Clinical preference**: Often favor higher recall (detect more lesions) at cost of some precision.

### Foreground Metrics

**Macro vs Micro:**
- **Macro**: Treats all classes equally (average of per-class metrics)
- **Micro**: Weighted by class frequency (aggregates all predictions)

Use macro when classes are important equally; micro when frequency matters.

## Advanced Analysis

### Analyzing Per-Case Performance

Identify challenging cases:

```bash
# Sort per-case CSV by Dice score
sort -t, -k5 -n test_evaluation_report_per_case.csv | head -10
```

Look for:
- Cases with low Dice scores (<0.5)
- High variance in performance
- Systematic failures (e.g., all CHO cases)

### Cross-Validation Analysis

From summary CSV:

1. **Check consistency**: Std Dev should be low (<0.02 for Dice)
2. **Identify outlier folds**: Folds with >2 std from mean
3. **Compare to ensemble**: Ensemble should outperform mean fold

### Statistical Significance

To compare two models:

```python
import pandas as pd
import scipy.stats as stats

# Load summaries
model_a = pd.read_csv('summary_model_a.csv')
model_b = pd.read_csv('summary_model_b.csv')

# Extract fold performances (excluding mean/std rows)
dice_a = model_a[model_a['Fold'].str.startswith('fold_')]['Metastatic_Dice']
dice_b = model_b[model_b['Fold'].str.startswith('fold_')]['Metastatic_Dice']

# Paired t-test (same folds)
t_stat, p_value = stats.ttest_rel(dice_a, dice_b)
print(f"p-value: {p_value:.4f}")
```

### Confusion Matrix Analysis

From the text report confusion matrix:

```
                 Pred Background  Pred Metastatic  Pred HCC  Pred CHO
True Background      31250251           795660      1199299    346823
True Metastatic       764565          6982162            0         0
True HCC              441293                0      4280676         0
True CHO              360906                0       227011   1629850
```

**Insights:**
- HCC→Background: 441,293 pixels (under-segmentation)
- Background→HCC: 1,199,299 pixels (over-segmentation)
- Metastatic has no confusion with HCC/CHO (good separation)

## Troubleshooting

### Error: "No prediction files found"

**Cause**: Predictions directory is empty or wrong format.

**Solution**:
```bash
# Check predictions exist
ls predictions/Dataset104_test/fold_0/*.png

# Verify PNG format (not NPZ or other)
file predictions/Dataset104_test/fold_0/*.png
```

### Error: "Prediction and label shapes don't match"

**Cause**: Different image sizes or preprocessing.

**Solution**:
- Ensure test images match label dimensions
- Check nnU-Net resampling settings

### Metrics seem incorrect

**Verification steps:**

1. Check confusion matrix makes sense
2. Verify Dice formula: `Dice = 2*TP/(2*TP + FP + FN)`
3. Compute manually for one class:

```python
import numpy as np
from PIL import Image

pred = np.array(Image.open('pred.png'))
label = np.array(Image.open('label.png'))

# For class 1 (Metastatic)
pred_binary = (pred == 1)
label_binary = (label == 1)

intersection = (pred_binary & label_binary).sum()
union = pred_binary.sum() + label_binary.sum()
dice = 2 * intersection / union

print(f"Dice: {dice:.4f}")
```

### Summary has NaN values

**Cause**: Missing metrics in some folds.

**Solution**:
- Check all fold JSON files exist
- Verify all folds completed successfully
- Ensure class labels are consistent

## Best Practices

1. **Always compute metrics** on a held-out test set
2. **Report mean ± std** across cross-validation folds
3. **Include ensemble performance** for final results
4. **Analyze confusion matrix** to understand failure modes
5. **Check per-case performance** for outliers
6. **Use Dice as primary metric** for segmentation
7. **Report multiple metrics** (Dice, precision, recall) for completeness

## Example Workflow

### Complete Evaluation Pipeline

```bash
# 1. Run inference with metrics
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/final_evaluation \
    --configuration 2d \
    --all_folds \
    --trainer nnUNetTrainer_500epochs \
    --compute_metrics \
    --labels_dir nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs \
    --generate_summary

# 2. Check summary
cat predictions/final_evaluation/metrics_summary.csv

# 3. Analyze per-case performance
python -c "
import pandas as pd
df = pd.read_csv('predictions/final_evaluation/ensemble/test_evaluation_report_per_case.csv')
print('Worst cases:')
print(df.nsmallest(5, 'Metastatic_Dice')[['case_id', 'Metastatic_Dice']])
"

# 4. Generate plots (optional)
# Use visualization tools to plot Dice scores, confusion matrices, etc.
```

## Related Documentation

- [Inference Guide](inference.md) - Running predictions
- [Training Guide](training.md) - Model training
- [Preprocessing Guide](preprocessing.md) - Data preparation

## Additional Resources

- Dice Score: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
- Jaccard Index: https://en.wikipedia.org/wiki/Jaccard_index
- Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
- Medical Image Segmentation Metrics: https://arxiv.org/abs/2106.05982
