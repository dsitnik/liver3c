# Classification Training Guide

**Patch-Based Histopathological Image Classification with timm Models**

This document describes the complete training procedure for patch-based classification as a comparison to nnU-Net segmentation, with reproducibility guarantees and publication-ready documentation.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Training Procedure](#training-procedure)
4. [Patch Extraction and Filtering](#patch-extraction-and-filtering)
5. [Models](#models)
6. [Reproducibility](#reproducibility)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Troubleshooting](#troubleshooting)
9. [For Publication](#for-publication)
10. [Advanced Usage](#advanced-usage)

---

## Overview

### Purpose

This classification approach serves as a **baseline comparison** to nnU-Net segmentation. Instead of pixel-level segmentation, we extract patches from histopathological images and classify them into 4 categories:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Background | Non-cancerous tissue |
| 1 | Metastatic | Metastatic liver cancer |
| 2 | HCC | Hepatocellular carcinoma |
| 3 | CHO | Cholangiocarcinoma |

### Key Features

**Full Reproducibility**
- Fixed random seeds (default: 42, per-fold variation)
- Deterministic CUDA operations
- Dataset-specific normalization from nnU-Net fingerprint
- Complete version tracking

**Scientific Rigor**
- Same train/val splits as nnU-Net (from `splits_final.json`)
- Case-level splitting (no data leakage)
- Non-overlapping validation patches
- Background class balancing

**Fair Comparison with nnU-Net**
- Uses same datasets (Dataset100-104)
- Uses same 10-fold cross-validation splits
- Uses same normalization statistics
- Comparable training procedures

**Publication Ready**
- Comprehensive metadata logging
- Per-fold confusion matrices
- Training summaries (JSON + TXT)
- TensorBoard integration

---

## Prerequisites

### System Requirements

**Hardware**:
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 3090 or better recommended)
- RAM: 32GB+ recommended (for image caching)
- Storage: 50GB+ free space

**Software**:
- Python 3.10+
- PyTorch 2.1.2+
- CUDA 11.8+ / CUDA 12.x
- timm (PyTorch Image Models)
- albumentations

### Installation

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies (if not already installed)
pip install timm albumentations

# Verify installation
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import albumentations; print(f'albumentations: {albumentations.__version__}')"
```

### Dataset Requirements

**IMPORTANT**: Classification training requires:

1. **nnU-Net raw data** in `nnUNet_raw/Dataset{ID}_{Name}/`
   - `imagesTr/` - Training images (3-channel PNGs)
   - `labelsTr/` - Training labels (single-channel PNGs)
   - `dataset.json` - Dataset metadata

2. **nnU-Net preprocessed data** in `nnUNet_preprocessed/Dataset{ID}_{Name}/`
   - `splits_final.json` - 10-fold CV splits
   - `dataset_fingerprint.json` - Normalization statistics

If you haven't run nnU-Net preprocessing yet:
```bash
python scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --plan_only
```

---

## Training Procedure

### Basic Training

#### Train Single Model (All Folds)

```bash
python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --model convnextv2
```

#### Train All 5 Models

```bash
python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --all
```

#### Running in Screen Session (Recommended)

```bash
# Single model
screen -dmS cls_convnextv2 bash -c "source .venv/bin/activate && \
    python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --model convnextv2 \
    2>&1 | tee classification_convnextv2.log; exec bash"

# All models
screen -dmS cls_all_models bash -c "source .venv/bin/activate && \
    python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --all \
    2>&1 | tee classification_all_models.log; exec bash"
```

**Monitor training**:
- Check screen sessions: `screen -ls`
- Attach to session: `screen -r cls_convnextv2`
- View log: `tail -f classification_training.log`
- Detach from screen: Press `Ctrl+A` then `D`

### Training Specific Folds

#### Single Fold

```bash
# Train only fold 0
python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --model convnextv2 \
    --fold 0
```

#### Fold Range

```bash
# Train folds 0-4 (first half)
python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --model swinv2 \
    --fold_range 0-4

# Train folds 5-9 (second half) on different GPU
CUDA_VISIBLE_DEVICES=1 python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --model swinv2 \
    --fold_range 5-9
```

### Resume Training

```bash
# Auto-detect and resume from checkpoint
python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --model convnextv2 \
    --resume

# Force restart (ignore existing checkpoints)
python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --dataset_name Liver1 \
    --model convnextv2 \
    --force_restart
```

### Training All Datasets

To train on all 5 dataset partitions (for complete comparison with nnU-Net):

```bash
# Loop through all datasets
for id in 100 101 102 103 104; do
    name="Liver$((id - 99))"
    screen -dmS "cls_${id}" bash -c "source .venv/bin/activate && \
        python scripts/classification/classification_training.py \
        --dataset_id ${id} \
        --dataset_name ${name} \
        --all \
        2>&1 | tee classification_${id}.log; exec bash"
done
```

---

## Patch Extraction and Filtering

### Patch Size

Default: **384x384 pixels**

This size was chosen to:
- Match input requirements of larger models (MaxViT, Swin-V2)
- Provide sufficient context for classification
- Balance between detail and computational cost

### Patch Sampling Strategy

**Training**: 50% overlapping grid
- Step size = patch_size / 2 = 192 pixels
- More samples for training
- Acceptable correlation within same case

**Validation**: Non-overlapping grid
- Step size = patch_size = 384 pixels
- Unbiased evaluation metrics
- No artificial inflation from correlated patches

### Patch Filtering Rules

Patches are classified based on their label content:

```
┌─────────────────────────────────────────────────────────────┐
│                    PATCH FILTERING LOGIC                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Cancer % = 0        →  KEEP as Background (class 0)       │
│                          (Important for learning normal)    │
│                                                             │
│  0 < Cancer % < 10%  →  DISCARD                            │
│                          (Ambiguous border patches)         │
│                                                             │
│  Cancer % >= 10%     →  KEEP as Cancer                     │
│                          (Assign dominant cancer class)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Rationale**:
- Pure background patches teach the model what healthy tissue looks like
- Borderline patches (1-9% cancer) are ambiguous and noisy
- 10% threshold ensures meaningful cancer content for classification

### Background Balancing

To prevent background class dominance:

```
max_background_patches = avg_cancer_class_count × max_background_ratio
```

Default `max_background_ratio = 2.0` means background patches are limited to 2x the average number of cancer patches per class.

**Example**:
- Metastatic patches: 500
- HCC patches: 600
- CHO patches: 400
- Average: 500
- Max background: 500 × 2 = 1000 (subsampled from original count)

---

## Models

### Available Models

Five state-of-the-art timm models optimized for image classification:

| Model ID | Full Name | Parameters | Input Size | Notes |
|----------|-----------|------------|------------|-------|
| `convnextv2` | ConvNeXt-V2-Base | 89M | 224-384 | Modern CNN, ImageNet-22k pretrained |
| `efficientnetv2` | EfficientNetV2-M | 54M | 384 | Excellent efficiency-accuracy tradeoff |
| `swinv2` | Swin Transformer V2-Base | 88M | 192→384 | Hierarchical vision transformer |
| `maxvit` | MaxViT-Base | 120M | 384 | Hybrid CNN-Transformer |
| `densenet` | DenseNet-161 | 29M | 224 | Proven for medical imaging |

### Model Selection Guidelines

**For best accuracy**: `swinv2` or `maxvit`
- State-of-the-art architectures
- Higher computational cost

**For efficiency**: `efficientnetv2`
- Best accuracy per FLOP
- Faster training

**For medical imaging baseline**: `densenet`
- Well-established in radiology
- Smaller model, faster training

**For general purpose**: `convnextv2`
- Modern CNN architecture
- Good balance of speed and accuracy

### Pretrained Weights

All models use ImageNet-21k/22k pretrained weights where available:

```python
MODELS = {
    'convnextv2': 'convnextv2_base.fcmae_ft_in22k_in1k',
    'efficientnetv2': 'tf_efficientnetv2_m.in21k_ft_in1k',
    'swinv2': 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
    'maxvit': 'maxvit_base_tf_384.in21k_ft_in1k',
    'densenet': 'densenet161.tv_in1k'
}
```

---

## Reproducibility

### Random Seed Control

**Base seed**: 42 (configurable via `--random_seed`)

**Per-fold seeding**:
```python
fold_seed = random_seed + fold  # e.g., fold 0 = 42, fold 1 = 43, ...
```

This ensures:
- Different but reproducible randomness per fold
- Consistent results when rerunning same fold
- Comparability across experiments

### What Gets Seeded

At the start of each fold:

1. **Python random**: `random.seed(fold_seed)`
2. **NumPy**: `np.random.seed(fold_seed)`
3. **PyTorch**: `torch.manual_seed(fold_seed)`
4. **CUDA**: `torch.cuda.manual_seed_all(fold_seed)`
5. **cuDNN**: Deterministic mode enabled

### Environment Variables

```bash
PYTHONHASHSEED=42
CUBLAS_WORKSPACE_CONFIG=:4096:8  # Deterministic CUDA operations
```

### Normalization Statistics

**Critical for fair comparison with nnU-Net!**

Normalization uses dataset-specific statistics from `dataset_fingerprint.json`:

```python
# Loaded automatically from nnU-Net preprocessing
mean = [channel0_mean/255, channel1_mean/255, channel2_mean/255]
std = [channel0_std/255, channel1_std/255, channel2_std/255]
```

This matches nnU-Net's preprocessing, ensuring:
- Same intensity normalization as segmentation
- Fair comparison between methods
- No train/test distribution shift

### Data Leakage Prevention

| Concern | Protection |
|---------|------------|
| Train/Val split | Case-level from `splits_final.json` |
| Patch correlation | Non-overlapping validation patches |
| Normalization | Global dataset stats (same as nnU-Net) |
| Background sampling | Fixed seed, training only |

---

## Monitoring and Logging

### Output Directory Structure

```
classification_results/
└── Dataset100_Liver1/
    └── convnextv2/
        ├── fold_0/
        │   ├── checkpoint_best.pth
        │   ├── checkpoint_final.pth
        │   ├── training_log.csv
        │   ├── confusion_matrix_best.json
        │   ├── confusion_matrix_final.json
        │   └── tensorboard/
        ├── fold_1/
        │   └── ...
        └── fold_9/
            └── ...

classification_metadata/
└── Dataset100_Liver1/
    └── 20250104_143052/              # Session ID
        ├── system_info.json          # Hardware specs
        ├── software_versions.json    # Package versions
        ├── dataset_info.json         # Dataset configuration
        ├── normalization_stats.json  # Mean/std used
        ├── training_summary.json     # Complete report
        └── training_summary.txt      # Human-readable summary
```

### Training Log CSV

Each fold creates `training_log.csv` with columns:

| Column | Description |
|--------|-------------|
| epoch | Training epoch (1-indexed) |
| train_loss | Cross-entropy loss on training set |
| train_acc | Training accuracy |
| val_loss | Validation loss |
| val_acc | Validation accuracy |
| val_f1 | Macro F1 score |
| lr | Current learning rate |
| time_elapsed | Epoch duration (seconds) |

### Confusion Matrix JSON

```json
{
  "epoch": 45,
  "class_names": ["background", "metastatic", "hcc", "cho"],
  "matrix": [[100, 5, 2, 1], [3, 85, 8, 4], ...],
  "per_class_metrics": {
    "background": {"precision": 0.92, "recall": 0.93, "f1": 0.92, "support": 108},
    "metastatic": {"precision": 0.85, "recall": 0.85, "f1": 0.85, "support": 100},
    ...
  }
}
```

### Real-Time Monitoring

```bash
# Follow training log
tail -f classification_training.log

# Watch GPU usage
watch -n 1 nvidia-smi

# View latest metrics
tail -20 classification_results/Dataset100_Liver1/convnextv2/fold_0/training_log.csv

# TensorBoard
tensorboard --logdir classification_results/Dataset100_Liver1/convnextv2/fold_0/tensorboard --port 6007
```

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:

**A. Reduce batch size**:
```bash
python scripts/classification/classification_training.py --dataset_id 100 --model convnextv2 --batch_size 16
```

**B. Use smaller model**:
```bash
# DenseNet is smallest
python scripts/classification/classification_training.py --dataset_id 100 --model densenet
```

**C. Use smaller patch size** (not recommended for fair comparison):
```bash
python scripts/classification/classification_training.py --dataset_id 100 --model convnextv2 --patch_size 224
```

#### Issue 2: Dataset Fingerprint Not Found

**Symptoms**:
```
WARNING - Dataset fingerprint not found: ...
WARNING - Falling back to simple [0,1] normalization
```

**Solution**: Run nnU-Net preprocessing first:
```bash
python scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --plan_only
```

#### Issue 3: No Cancer Patches Found

**Symptoms**:
```
WARNING - No cancer patches found!
```

**Cause**: Labels might use different class values than expected (0-3).

**Solution**: Verify label values:
```python
from PIL import Image
import numpy as np
label = np.array(Image.open("nnUNet_raw/Dataset100_Liver1/labelsTr/metastatic_1.png"))
print(f"Unique values: {np.unique(label)}")  # Should be [0, 1] or [0, 2] or [0, 3]
```

#### Issue 4: Training Too Slow

**Symptoms**: Each epoch takes >10 minutes

**Solutions**:

**A. Increase num_workers**:
Edit `classification_config.json`:
```json
{
    "num_workers": 8
}
```

**B. Reduce image cache size** (if RAM limited):
Edit `scripts/classification/classification_training.py`:
```python
_cache_max_size = 100  # Reduce from 200
```

**C. Use faster model**:
```bash
python scripts/classification/classification_training.py --model densenet  # Smallest/fastest
```

#### Issue 5: Resume Not Working

**Symptoms**: Training restarts from epoch 0 despite `--resume`

**Check**:
```bash
# Verify checkpoints exist
ls classification_results/Dataset100_Liver1/convnextv2/fold_0/checkpoint_*.pth
```

**If missing**: Previous training may have crashed before first checkpoint. Use `--force_restart`.

---

## For Publication

### Methods Section Template

```markdown
## Patch-Based Classification

### Overview

As a comparison to pixel-level segmentation, we implemented patch-based
classification using state-of-the-art convolutional and transformer
architectures.

### Patch Extraction

Images were divided into 384×384 pixel patches using a sliding window
approach. During training, patches were extracted with 50% overlap to
increase sample size. During validation, non-overlapping patches were
used to prevent metric inflation from correlated samples.

Patches were filtered based on cancer content:
- Pure background patches (0% cancer) were retained to learn normal tissue
- Patches with 1-9% cancer were discarded as ambiguous
- Patches with ≥10% cancer were labeled with the dominant cancer class

To address class imbalance, background patches were subsampled to at most
2× the average cancer class count.

### Models

We evaluated five architectures pretrained on ImageNet-21k/22k:
1. ConvNeXt-V2-Base (89M parameters)
2. EfficientNetV2-M (54M parameters)
3. Swin Transformer V2-Base (88M parameters)
4. MaxViT-Base (120M parameters)
5. DenseNet-161 (29M parameters)

### Training Configuration

- Input: 384×384×3 patches normalized with dataset-specific statistics
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: Cosine annealing
- Loss: Cross-entropy with inverse-frequency class weights
- Batch size: 32
- Epochs: 100 (early stopping, patience=20)
- Gradient clipping: max_norm=1.0

### Data Augmentation

Training augmentation included:
- Random 90° rotations, horizontal/vertical flips
- Shift-scale-rotate (±10% shift, ±15% scale, ±45° rotation)
- Elastic transform, grid distortion, optical distortion
- Gaussian noise, ISO noise
- Random brightness/contrast adjustment
- Gaussian blur

### Cross-Validation

We used the same 10-fold cross-validation splits as nnU-Net segmentation
to ensure fair comparison. The same normalization statistics (per-channel
mean and standard deviation computed by nnU-Net preprocessing) were used.

### Evaluation

Models were evaluated using:
- Overall accuracy
- Per-class accuracy
- Macro F1 score
- Confusion matrices

The checkpoint with the best validation F1 score was selected for each fold.
```

### Supplementary Materials

Include:
1. `training_summary.json` - Complete training record
2. `system_info.json` - Hardware specifications
3. `software_versions.json` - Package versions
4. `normalization_stats.json` - Exact mean/std used
5. `confusion_matrix_best.json` - Per-fold confusion matrices
6. `scripts/classification/classification_training.py` - Training script
7. `classification_config.json` - Configuration file

### Reporting Checklist

Before submission, verify you have:

- [ ] Documented patch extraction procedure
- [ ] Reported patch filtering thresholds
- [ ] Explained background balancing
- [ ] Listed all model architectures and parameters
- [ ] Specified training hyperparameters
- [ ] Documented data augmentation
- [ ] Stated normalization source (nnU-Net fingerprint)
- [ ] Reported cross-validation strategy
- [ ] Provided per-fold metrics (mean ± std)
- [ ] Included confusion matrices
- [ ] Made training scripts available

---

## Advanced Usage

### Custom Configuration

Edit `classification_config.json`:

```json
{
    "patch_size": 384,
    "batch_size": 32,
    "max_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "min_cancer_threshold": 0.10,
    "max_background_ratio": 2.0,
    "grad_clip_norm": 1.0,
    "num_workers": 4,
    "early_stopping_patience": 20,
    "continue_on_error": false,
    "device": {
        "use_cuda": true,
        "cuda_device": 0
    }
}
```

### Command-Line Overrides

Most config parameters can be overridden:

```bash
python scripts/classification/classification_training.py \
    --dataset_id 100 \
    --model convnextv2 \
    --patch_size 224 \
    --batch_size 64 \
    --epochs 50 \
    --learning_rate 0.001 \
    --min_cancer_threshold 0.15
```

### Multi-GPU Training

Run different models/folds on different GPUs:

```bash
# GPU 0: ConvNeXt folds 0-4
CUDA_VISIBLE_DEVICES=0 python scripts/classification/classification_training.py \
    --dataset_id 100 --model convnextv2 --fold_range 0-4 &

# GPU 1: ConvNeXt folds 5-9
CUDA_VISIBLE_DEVICES=1 python scripts/classification/classification_training.py \
    --dataset_id 100 --model convnextv2 --fold_range 5-9 &

# GPU 2: Swin all folds
CUDA_VISIBLE_DEVICES=2 python scripts/classification/classification_training.py \
    --dataset_id 100 --model swinv2 &
```

### Adding Custom Models

To add a new timm model, edit `scripts/classification/classification_training.py`:

```python
MODELS = {
    'convnextv2': 'convnextv2_base.fcmae_ft_in22k_in1k',
    'efficientnetv2': 'tf_efficientnetv2_m.in21k_ft_in1k',
    'swinv2': 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
    'maxvit': 'maxvit_base_tf_384.in21k_ft_in1k',
    'densenet': 'densenet161.tv_in1k',
    # Add your model:
    'resnet50': 'resnet50.a1_in1k',
}
```

Then use: `--model resnet50`

### Inference (After Training)

Create `scripts/classification/classification_inference.py` for test set evaluation (TODO):

```bash
python scripts/classification/classification_inference.py \
    --dataset_id 100 \
    --model convnextv2 \
    --all_folds \
    --compute_metrics
```

---

## Quick Reference

### Command Cheat Sheet

```bash
# Train single model, all folds
python scripts/classification/classification_training.py --dataset_id 100 --dataset_name Liver1 --model convnextv2

# Train all 5 models
python scripts/classification/classification_training.py --dataset_id 100 --dataset_name Liver1 --all

# Train specific fold
python scripts/classification/classification_training.py --dataset_id 100 --model convnextv2 --fold 0

# Train fold range
python scripts/classification/classification_training.py --dataset_id 100 --model swinv2 --fold_range 0-4

# Resume training
python scripts/classification/classification_training.py --dataset_id 100 --model convnextv2 --resume

# Force restart
python scripts/classification/classification_training.py --dataset_id 100 --model convnextv2 --force_restart

# Custom settings
python scripts/classification/classification_training.py --dataset_id 100 --model densenet \
    --batch_size 64 --epochs 50 --learning_rate 0.001
```

### File Locations

```bash
# Results (models and logs)
classification_results/Dataset{ID}_{Name}/{model}/fold_{N}/

# Metadata (per session)
classification_metadata/Dataset{ID}_{Name}/{session_id}/

# Main log file
classification_training.log

# Config file
classification_config.json
```

### Expected Training Time

**Hardware**: NVIDIA RTX 3090 (24GB VRAM)

| Model | Per Fold | All 10 Folds |
|-------|----------|--------------|
| DenseNet | 1-2 hours | 10-20 hours |
| EfficientNetV2 | 2-3 hours | 20-30 hours |
| ConvNeXt-V2 | 2-3 hours | 20-30 hours |
| Swin-V2 | 3-4 hours | 30-40 hours |
| MaxViT | 3-4 hours | 30-40 hours |
| **All 5 models** | - | **110-160 hours** |

---

## References

### Related Documentation

- `segmentation_training.md` - nnU-Net training guide
- `preprocessing.md` - Dataset preparation
- `metrics.md` - Evaluation metrics

### Scripts

- `scripts/classification/classification_training.py` - Main training script
- `classification_config.json` - Configuration file
- `scripts/segmentation/nnunet_training.py` - nnU-Net training (for comparison)

### External Resources

- [timm Documentation](https://huggingface.co/docs/timm)
- [timm Model Zoo](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/docs/)

---

**Last Updated**: January 2025
**Version**: 1.0
**Author**: Liver Cancer Classification Project
