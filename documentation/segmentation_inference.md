# nnU-Net Inference Guide

This document provides comprehensive instructions for running inference and prediction on test sets using trained nnU-Net models.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Inference Modes](#inference-modes)
- [Command-Line Options](#command-line-options)
- [Examples](#examples)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## Overview

The `scripts/segmentation/nnunet_inference.py` script provides a comprehensive interface for running predictions on test data using trained nnU-Net models. It supports:

- Single fold inference
- Multi-fold ensemble inference
- Automatic metrics computation
- Visualization generation
- MATLAB export
- Custom trainer support (e.g., 500-epoch models)

## Quick Start

**IMPORTANT**: All commands require activating the virtual environment first:
```bash
source .venv/bin/activate
```

### Basic Inference with Ensemble

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/Dataset104_test \
    --configuration 2d \
    --all_folds \
    --trainer nnUNetTrainer_500epochs \
    --compute_metrics \
    --labels_dir nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs
```

This will:
1. Run inference on each fold individually (fold_0 through fold_9)
2. Create ensemble predictions from all folds
3. Compute test metrics for each fold and ensemble
4. Save predictions and metrics to the output directory

### Running in Screen Session (Recommended for Long Runs)

For long-running inference tasks, use a detached screen session:

```bash
# Basic screen session (no logging)
screen -dmS nnunet_inference bash -c "source .venv/bin/activate && python scripts/segmentation/nnunet_inference.py --dataset_id 104 --dataset_name LiverCancerStratified --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs --output_dir predictions/complete --configuration 2d --all_folds --trainer nnUNetTrainer_500epochs --compute_metrics --labels_dir nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs; exec bash"

# With all features (visualizations, MATLAB export, summary)
screen -dmS nnunet_inference bash -c "source .venv/bin/activate && python scripts/segmentation/nnunet_inference.py --dataset_id 104 --dataset_name LiverCancerStratified --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs --output_dir predictions/final --configuration 2d --all_folds --trainer nnUNetTrainer_500epochs --compute_metrics --labels_dir nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs --visualize --export_mat --generate_summary; exec bash"

# With output logging to file (recommended)
screen -dmS nnunet_inference bash -c "source .venv/bin/activate && python scripts/segmentation/nnunet_inference.py --dataset_id 104 --dataset_name LiverCancerStratified --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs --output_dir predictions/complete --configuration 2d --all_folds --trainer nnUNetTrainer_500epochs --compute_metrics --labels_dir nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs 2>&1 | tee inference_output.log; exec bash"
```

**Screen session commands:**
```bash
screen -r nnunet_inference  # Attach to session to view progress
# Press Ctrl+A, then D to detach from session
screen -ls                  # List all active screen sessions
tail -f inference_output.log  # Monitor log file (if using logging option)
```

## Inference Modes

### 1. Single Fold Inference

Run inference using a specific trained fold:

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/fold0_only \
    --configuration 2d \
    --fold 0 \
    --trainer nnUNetTrainer_500epochs
```

**Use case**: Quick testing, debugging, or when you only have one trained fold.

### 2. Multi-Fold Ensemble

Run inference with multiple folds and create ensemble predictions:

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/ensemble \
    --configuration 2d \
    --folds 0 1 2 3 4 \
    --trainer nnUNetTrainer_500epochs
```

**Use case**: When you want to use specific folds (e.g., only trained folds).

### 3. All Folds + Ensemble (Recommended)

Run inference on all folds individually, then create ensemble:

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/complete \
    --configuration 2d \
    --all_folds \
    --trainer nnUNetTrainer_500epochs \
    --compute_metrics \
    --labels_dir nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs
```

**Use case**: Complete evaluation - provides per-fold and ensemble metrics.

## Command-Line Options

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset_id` | Dataset ID (3-digit number) | `104` |
| `--input_dir` | Directory with test images | `nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs` |
| `--output_dir` | Directory to save predictions | `predictions/test` |

### Model Selection

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--dataset_name` | Dataset name | `LiverCancer` | `LiverCancerStratified` |
| `--configuration` | nnU-Net configuration | Auto-detect | `2d`, `3d_fullres` |
| `--trainer` | Custom trainer class | `nnUNetTrainer` | `nnUNetTrainer_500epochs` |
| `--checkpoint` | Checkpoint to use | `checkpoint_best.pth` | `checkpoint_final.pth` |

### Fold Selection

| Argument | Description | Example |
|----------|-------------|---------|
| `--fold` | Use specific fold | `--fold 0` |
| `--folds` | Use multiple folds | `--folds 0 1 2 3 4` |
| `--all_folds` | Use all trained folds | `--all_folds` |

### Inference Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--disable_tta` | Disable test-time augmentation | Enabled |
| `--step_size` | Sliding window step size | `0.5` |
| `--save_probabilities` | Save softmax probabilities | Not saved |
| `--cpu` | Use CPU instead of GPU | Uses GPU |

### Post-Processing

| Argument | Description |
|----------|-------------|
| `--compute_metrics` | Compute test metrics after inference |
| `--labels_dir` | Ground truth labels directory (required for metrics) |
| `--visualize` | Create RGB visualizations |
| `--export_mat` | Export to MATLAB .mat format |
| `--generate_summary` | Generate summary CSV across folds |

## Examples

### Example 1: Complete Evaluation with All Features

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/complete_evaluation \
    --configuration 2d \
    --all_folds \
    --trainer nnUNetTrainer_500epochs \
    --compute_metrics \
    --labels_dir nnUNet_raw/Dataset104_LiverCancerStratified/labelsTs \
    --visualize \
    --generate_summary
```

**Output:**
- `predictions/complete_evaluation/fold_0/` - Fold 0 predictions + metrics
- `predictions/complete_evaluation/fold_1/` - Fold 1 predictions + metrics
- ... (folds 2-9)
- `predictions/complete_evaluation/ensemble/` - Ensemble predictions + metrics
- `predictions/complete_evaluation/metrics_summary.csv` - Summary across folds

### Example 2: Quick Single Fold Test

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/quick_test \
    --configuration 2d \
    --fold 0 \
    --trainer nnUNetTrainer_500epochs
```

### Example 3: Ensemble Only (No Individual Folds)

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/ensemble_only \
    --configuration 2d \
    --folds 0 1 2 3 4 5 6 7 8 9 \
    --trainer nnUNetTrainer_500epochs
```

### Example 4: With Visualizations and MATLAB Export

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/with_viz \
    --configuration 2d \
    --all_folds \
    --trainer nnUNetTrainer_500epochs \
    --visualize \
    --export_mat
```

## Output Files

### Directory Structure

```
predictions/Dataset104_test/
├── fold_0/
│   ├── *.png                                    # Predicted segmentation masks
│   ├── test_evaluation_report.txt               # Human-readable metrics
│   ├── test_evaluation_report.json              # Machine-readable metrics
│   ├── test_evaluation_report_per_case.csv      # Per-case metrics
│   ├── test_evaluation_report_aggregate.csv     # Aggregate metrics
│   ├── visualizations/                          # RGB visualizations (if --visualize)
│   └── mat/                                     # MATLAB files (if --export_mat)
├── fold_1/
│   └── ... (same structure)
├── ... (folds 2-9)
├── ensemble/
│   ├── *.png                                    # Ensemble predictions
│   ├── test_evaluation_report.txt
│   ├── test_evaluation_report.json
│   ├── test_evaluation_report_per_case.csv
│   └── test_evaluation_report_aggregate.csv
├── metrics_summary.csv                          # Summary across folds (if --generate_summary)
└── metrics_summary_transposed.csv               # Transposed summary
```

### File Descriptions

#### Predictions
- **`*.png`**: Predicted segmentation masks (same filename as input)
- **Format**: PNG images with pixel values 0-3 (background, metastatic, HCC, CHO)

#### Metrics (if `--compute_metrics`)
- **`test_evaluation_report.txt`**: Human-readable text report with all metrics
- **`test_evaluation_report.json`**: JSON format for programmatic access
- **`test_evaluation_report_per_case.csv`**: Metrics for each test image
- **`test_evaluation_report_aggregate.csv`**: Overall aggregate metrics

#### Summary (if `--generate_summary`)
- **`metrics_summary.csv`**: Per-fold metrics + mean ± stddev
- **`metrics_summary_transposed.csv`**: Transposed for easier reading

## Ensemble Inference Explained

### How Ensemble Works

nnU-Net ensemble combines predictions from multiple folds by:

1. **Loading all fold models** (e.g., folds 0-9)
2. **Running inference** on each test image with all models
3. **Averaging softmax probabilities** across all models
4. **Taking argmax** to get final class prediction

**Important**: Ensemble is NOT simply averaging hard predictions. It averages probability maps before making final predictions, which is more robust.

### When to Use Ensemble

✅ **Use ensemble when:**
- You have multiple trained folds available
- You want the best possible performance
- You're reporting final results

❌ **Don't use ensemble when:**
- You only have one fold trained
- You're doing quick tests/debugging
- You want to compare individual fold performance

## Using Custom Trainers

If you trained models with a custom trainer (e.g., 500 epochs instead of default 1000):

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --trainer nnUNetTrainer_500epochs \
    ... (other arguments)
```

**Important**: The `--trainer` argument must match the trainer used during training. nnU-Net will look for checkpoints in:
```
nnUNet_results/Dataset104_LiverCancerStratified/nnUNetTrainer_500epochs__nnUNetPlans__2d/fold_X/
```

## Troubleshooting

### Error: "No trained folds found"

**Cause**: Trainer class doesn't match the trained models.

**Solution**:
- Check which trainer was used during training
- Verify checkpoint paths exist: `ls nnUNet_results/Dataset104_*/nnUNetTrainer*`
- Add correct `--trainer` argument

### Error: "checkpoint_best.pth not found"

**Cause**: Training didn't complete or checkpoint was deleted.

**Solution**:
```bash
# Check available checkpoints
ls nnUNet_results/Dataset104_*/nnUNetTrainer*__2d/fold_0/

# Use different checkpoint if needed
python scripts/segmentation/nnunet_inference.py --checkpoint checkpoint_final.pth ...
```

### Slow Inference

**Solution**:
- Increase step size: `--step_size 1.0` (faster but less accurate)
- Disable TTA: `--disable_tta`
- Use fewer folds for quick testing

### Out of Memory

**Solution**:
- Use CPU: `--cpu`
- Reduce batch size by decreasing step size
- Run folds sequentially instead of `--all_folds`

## Best Practices

1. **Always use `checkpoint_best.pth`** (default) for final evaluation
2. **Use ensemble** with all folds for best performance
3. **Compute metrics** to track performance: `--compute_metrics`
4. **Generate summary** for cross-validation: `--generate_summary`
5. **Keep test data separate** - never use images seen during training/validation

## Related Documentation

- [Metrics Calculation Guide](metrics.md) - Detailed metrics computation
- [Training Guide](training.md) - Model training instructions
- [Preprocessing Guide](preprocessing.md) - Data preparation

## Additional Resources

- nnU-Net Documentation: https://github.com/MIC-DKFZ/nnUNet
- Test-Time Augmentation: Mirroring along axes for robust predictions
- Sliding Window: Overlapping patches for inference on large images
