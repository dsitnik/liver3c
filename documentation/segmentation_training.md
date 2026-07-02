# nnUNet Training Guide

**Scientifically Rigorous Deep Learning for Medical Image Segmentation**

This document describes the complete training procedure with reproducibility guarantees, comprehensive monitoring, and publication-ready documentation.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Training Procedure](#training-procedure)
4. [Reproducibility](#reproducibility)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Troubleshooting](#troubleshooting)
7. [For Publication](#for-publication)
8. [Advanced Usage](#advanced-usage)

---

## Overview

### Key Features

Our training script (`scripts/segmentation/nnunet_training.py`) ensures:

✅ **Full Reproducibility**
- Fixed random seeds (default: 42)
- Deterministic CUDA operations
- Complete version tracking
- System configuration logging

✅ **Scientific Rigor**
- Proper train/test separation
- Patient-level cross-validation
- No data leakage
- Comprehensive metadata logging

✅ **Production Ready**
- Robust error handling
- GPU assignment management
- Resume capability
- Per-fold tracking

✅ **Publication Ready**
- Complete training documentation
- System specifications
- Hyperparameter logging
- Timing statistics

---

## Prerequisites

### System Requirements

**Hardware**:
- GPU: NVIDIA GPU with 11GB+ VRAM (RTX 2080 Ti or better)
- RAM: 32GB+ recommended
- Storage: 100GB+ free space

**Software**:
- Python 3.10+
- PyTorch 2.1.2+
- CUDA 11.8+ / CUDA 12.x
- nnUNet v2.x

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import nnunetv2; print('nnUNet: Installed')"
nnUNetv2_train --help
```

### Dataset Preparation

**IMPORTANT**: Complete preprocessing first!

```bash
# 1. Convert dataset (creates stratified CV splits)
python scripts/preprocessing/nnunet_conversion_with_partitions.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified

# 2. Verify no data leakage
python scripts/preprocessing/verify_partitions.py --dataset_id 104 --dataset_name LiverCancerStratified
python scripts/preprocessing/verify_partitions.py --dataset_id 104 --dataset_name LiverCancerStratified
```

See `preprocessing.md` for complete details.

---

## Training Procedure

### Basic Training

#### Default Training (1000 epochs)

**Complete workflow** (preprocessing + training):

```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --random_seed 42
```

#### Custom Epoch Training (500 epochs)

For faster training cycles, use the custom 500-epoch trainer:

```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --config train_config.json \
    --trainer nnUNetTrainer_500epochs \
    --random_seed 42
```

**Note**: The `--trainer` flag specifies a custom trainer class that trains for 500 epochs instead of the default 1000. The custom trainer (`nnUNetTrainer_500epochs.py`) must be in the project root directory.

#### Running in Screen Session (Recommended)

**For default 1000 epochs**:

```bash
screen -dmS nnunet_training_104 bash -c "source .venv/bin/activate && \
    python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --config train_config.json \
    --configurations 2d \
    --random_seed 42"
```

**For 500 epochs** (faster training):

```bash
screen -dmS nnunet_training_500ep bash -c "source .venv/bin/activate && \
    python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --config train_config.json \
    --trainer nnUNetTrainer_500epochs \
    --random_seed 42"
```

**Monitor training**:
- Check screen sessions: `screen -ls`
- Attach to session: `screen -r nnunet_training_104`
- View log: `tail -f nnunet_training.log`
- Detach from screen: Press `Ctrl+A` then `D`

This will:
1. Validate dataset integrity
2. Copy custom CV splits
3. Run nnUNet preprocessing (training data only)
4. Train all configurations (2D, 3D_fullres)
5. Train all 10 folds per configuration
6. Log comprehensive metadata

### Step-by-Step Execution

#### 1. Dataset Validation Only

```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --validate_only
```

Checks:
- Dataset structure
- Required files exist
- dataset.json format
- Saves validation results to metadata

#### 2. Preprocessing Only

```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --plan_only
```

Performs:
- Dataset validation
- Custom splits copy (BEFORE preprocessing)
- nnUNet planning and preprocessing
- Does NOT start training

**Verify after preprocessing**:
```bash
python scripts/preprocessing/verify_partitions.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified
```

Expected: ✅ No test set leakage detected

#### 3. Training Only (Skip Preprocessing)

```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --train_only
```

Use when:
- Preprocessing already completed
- Resuming failed training
- Testing different configurations

### Training Specific Folds

#### Single Fold (Default Trainer)

```bash
# Train only fold 0 with default 1000 epochs
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --fold 0 \
    --random_seed 42
```

#### Single Fold (500 Epochs)

```bash
# Train only fold 0 with 500 epochs
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --config train_config.json \
    --trainer nnUNetTrainer_500epochs \
    --fold 0 \
    --random_seed 42 \
    --train_only
```

**In screen session**:

```bash
screen -dmS nnunet_fold0_500ep bash -c "source .venv/bin/activate && \
    python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --config train_config.json \
    --trainer nnUNetTrainer_500epochs \
    --fold 0 \
    --random_seed 42 \
    --train_only"
```

**Use cases:**
- Testing training pipeline
- GPU memory testing
- Quick validation
- Training remaining fold after others complete

#### Fold Range (Default Trainer)

```bash
# Train folds 0-4 (first half)
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --fold_range 0-4 \
    --random_seed 42

# Train folds 5-9 (second half) - can run in parallel on different GPU
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --fold_range 5-9 \
    --random_seed 42
```

#### Fold Range (500 Epochs)

```bash
# Train folds 1-9 with 500 epochs (assuming fold 0 already trained)
screen -dmS nnunet_folds1-9_500ep bash -c "source .venv/bin/activate && \
    python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --config train_config.json \
    --trainer nnUNetTrainer_500epochs \
    --fold_range 1-9 \
    --random_seed 42"
```

**Use cases:**
- Multi-GPU training (run ranges on different GPUs)
- Interrupted training
- Distributed computation
- Training remaining folds after some complete

### Specific Configuration

```bash
# Train only 2D configuration
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --configurations 2d

# Train both 2D and 3D
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --configurations 2d 3d_fullres
```

---

## Reproducibility

### Random Seed Control

**Default seed: 42** (matches preprocessing)

```bash
# Use custom seed
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --random_seed 12345
```

### What Gets Seeded

1. **Python random module**: `random.seed(42)`
2. **NumPy**: `np.random.seed(42)`
3. **Environment**: `PYTHONHASHSEED=42`
4. **CUDA**: `CUBLAS_WORKSPACE_CONFIG=:4096:8` (deterministic ops)

### nnUNet Internal Seeds

nnUNet v2 handles additional seeding internally:
- PyTorch RNG
- DataLoader workers
- Augmentation pipeline

### Limitations

**Fully deterministic results require**:
- Same hardware (GPU model)
- Same CUDA version
- Same PyTorch version
- Same nnUNet version

**Near-deterministic** (expected):
- Training curves match closely
- Final metrics within ±0.5% Dice

**Non-deterministic elements**:
- Exact GPU floating-point operations
- Multi-GPU synchronization
- Some CUDA kernels

### Verification

Compare two runs:
```bash
# Run 1
python scripts/segmentation/nnunet_training.py --dataset_id 104 --random_seed 42

# Run 2 (same seed)
python scripts/segmentation/nnunet_training.py --dataset_id 104 --random_seed 42

# Compare results
diff -r nnUNet_results/Dataset104_*/fold_0/validation_raw/ \
         nnUNet_results_run2/Dataset104_*/fold_0/validation_raw/
```

Expected: Identical or nearly identical validation metrics

---

## Monitoring and Logging

### Training Output

During training, the script logs:

```
================================================================================
STARTING TRAINING
Configuration: 2d
Fold:          0
Device:        cuda
GPU ID:        0
================================================================================
Running: Training 2d fold 0
Command: nnUNetv2_train 104 2d 0 --npz -c -device cuda
================================================================================
Training log: training_metadata/Dataset104_LiverCancerStratified/20250129_143052/training_2d_fold0.log
```

### Log Locations

All training metadata saved to:
```
training_metadata/
└── Dataset104_LiverCancerStratified/
    └── 20250129_143052/                    # Session ID (timestamp)
        ├── system_info.json                # Hardware specs
        ├── software_versions.json          # Package versions
        ├── dataset_info.json               # Dataset configuration
        ├── splits_final.json               # CV splits used
        ├── preprocessing.log               # Preprocessing output
        ├── training_2d_fold0.log           # Per-fold training logs
        ├── training_2d_fold1.log
        ├── ...
        ├── training_summary.json           # Complete training report
        └── training_summary.txt            # Human-readable summary
```

### Real-Time Monitoring

#### Watch Training Progress

```bash
# Follow training log
tail -f training_metadata/Dataset104_*/20250129_143052/training_2d_fold0.log

# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor disk usage
df -h nnUNet_results/
```

#### Check Training Status

```bash
# View summary
cat training_metadata/Dataset104_*/20250129_143052/training_summary.txt

# Check per-fold status
python -c "
import json
with open('training_metadata/Dataset104_LiverCancerStratified/20250129_143052/training_summary.json') as f:
    summary = json.load(f)
for fold, result in summary['fold_results'].items():
    print(f'{fold:30s} {result[\"status\"]:10s} {result[\"elapsed_time\"]/3600:.2f}h')
"
```

### nnUNet Progress Files

nnUNet creates progress tracking files:

```bash
# Check current epoch
cat nnUNet_results/Dataset104_*/nnUNetTrainer__nnUNetPlans__2d/fold_0/progress.png
# (Or view progress.png image)

# Check validation metrics
cat nnUNet_results/Dataset104_*/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation_raw/summary.json
```

### TensorBoard (Optional)

nnUNet logs to TensorBoard automatically:

```bash
tensorboard --logdir nnUNet_results/Dataset104_LiverCancerStratified/ --port 6006
```

View at: http://localhost:6006

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:

**A. Reduce batch size** (in nnUNet automatically, but can force):
```bash
# nnUNet auto-adjusts batch size, but you can monitor:
watch -n 1 nvidia-smi
```

**B. Train 2D only** (lower memory):
```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --configurations 2d
```

**C. Use smaller GPU**: nnUNet v2 adapts to available memory

**D. Close other GPU processes**:
```bash
# Check what's using GPU
nvidia-smi

# Kill if needed
kill <PID>
```

#### Issue 2: Training Hangs or Freezes

**Symptoms**: No output for >10 minutes, GPU utilization 0%

**Solutions**:

**A. Check logs**:
```bash
tail -100 training_metadata/Dataset104_*/*/training_2d_fold0.log
```

**B. Check disk space**:
```bash
df -h nnUNet_results/
```

**C. Kill and resume**:
```bash
# Kill training
pkill -f nnUNetv2_train

# Resume with --resume flag
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --resume
```

#### Issue 3: "Splits file not found"

**Problem**: Preprocessing didn't find custom splits

**Solution**:
```bash
# Check splits exist
ls nnUNet_raw/Dataset104_*/splits_final.json

# If missing, rerun conversion:
python scripts/preprocessing/nnunet_conversion_with_partitions.py --dataset_id 104

# Then rerun training
python scripts/segmentation/nnunet_training.py --dataset_id 104 --train_only
```

#### Issue 4: GPU Assignment Not Working

**Problem**: All folds use same GPU despite configuration

**Solution**: Our `scripts/segmentation/nnunet_training.py` fixed this bug!

If still issues:
```bash
# Manual GPU selection
CUDA_VISIBLE_DEVICES=0 python scripts/segmentation/nnunet_training.py --dataset_id 104 --fold 0
CUDA_VISIBLE_DEVICES=1 python scripts/segmentation/nnunet_training.py --dataset_id 104 --fold 1
```

#### Issue 5: Resume Not Working

**Symptoms**: Training restarts from epoch 0 despite `--resume`

**Solutions**:

**A. Check checkpoints exist**:
```bash
ls nnUNet_results/Dataset104_*/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_*.pth
```

**B. Use explicit resume**:
```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --resume  # Explicitly enable
```

**C. Force restart if needed**:
```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --force_restart  # Ignore checkpoints
```

---

## For Publication

### Methods Section Template

```markdown
## Model Training

### Training Procedure

We used nnU-Net v2.x (Isensee et al., 2021) for automated architecture
configuration and training. nnU-Net automatically determines optimal
hyperparameters based on dataset properties, including network topology,
batch size, patch size, and learning rate schedule.

Training was performed using 10-fold cross-validation with patient-level
splitting. Each fold was trained independently using the nnUNet default
settings:
- Optimizer: SGD with momentum 0.99
- Learning rate: 0.01 with polynomial decay (exponent 0.9)
- Loss function: Combined Dice + Cross-Entropy loss
- Data augmentation: Rotation, scaling, elastic deformation, intensity shifts
- Epochs: 1000 with early stopping (50 epochs patience)
- Batch size: Auto-determined per GPU (typically 2)

We trained both 2D and 3D full resolution configurations. For 2D, axial
slices were processed independently. For 3D, full volumetric patches were
used.

### Computational Resources

Training was performed on [Hardware from system_info.json]:
- GPU: [GPU name, VRAM]
- CPU: [CPU cores]
- RAM: [RAM size]
- OS: [OS version]

Total training time was [X] hours for all configurations and folds
([Y] hours per fold on average). Detailed per-fold timing is provided
in supplementary materials.

### Software Environment

All software versions were logged for reproducibility:
- nnU-Net: v[version]
- PyTorch: v[version]
- CUDA: v[version]
- Python: v[version]

Complete software environment details are provided in supplementary
materials (software_versions.json).

### Reproducibility

To ensure reproducible results, we:
1. Set random seed to 42 for all random number generators
2. Enabled deterministic CUDA operations (CUBLAS_WORKSPACE_CONFIG=:4096:8)
3. Used fixed cross-validation splits (provided in supplementary materials)
4. Logged all training hyperparameters and system configuration

Training scripts, configuration files, and cross-validation splits are
available at [GitHub URL] and in supplementary materials.

### Model Selection

For each configuration, we selected the model checkpoint with the best
validation Dice score during training. Final test set evaluation used
an ensemble of all 10 fold models, averaging softmax probabilities
before argmax prediction.
```

### Supplementary Materials

Include in your paper's supplementary materials:

**Required**:
1. `training_summary.json` - Complete training record
2. `system_info.json` - Hardware specifications
3. `software_versions.json` - Package versions
4. `splits_final.json` - Exact CV splits used
5. Training scripts - `scripts/segmentation/nnunet_training.py`

**Recommended**:
6. Per-fold training logs - `training_2d_fold*.log`
7. Training hyperparameters - From nnUNetPlans.json
8. Validation curves - From TensorBoard logs

### Reporting Checklist

Before submission, verify you have:

- [ ] Documented train/test split procedure
- [ ] Reported cross-validation strategy
- [ ] Listed all hyperparameters (or stated "nnUNet defaults")
- [ ] Specified training time and hardware
- [ ] Provided software versions
- [ ] Documented reproducibility measures
- [ ] Made training scripts available
- [ ] Reported per-fold metrics (mean ± std)
- [ ] Stated model selection criterion
- [ ] Described ensemble methodology

---

## Advanced Usage

### Custom Trainers

#### nnUNetTrainer_500epochs

The project includes a custom trainer that trains for 500 epochs instead of the default 1000 epochs. This is useful for:
- Faster experimentation and iteration
- Resource-constrained environments
- Preliminary results before full training

**Implementation** (`nnUNetTrainer_500epochs.py`):

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_500epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json,
                        unpack_dataset, device)

        # Override default epoch settings
        self.num_epochs = 500
        self.max_num_epochs = 500
```

**Key Points:**
- Inherits all default nnU-Net behavior
- Only modifies epoch count
- Must be in project root or Python path
- Results saved in `nnUNetTrainer_500epochs__nnUNetPlans__2d/` directory

**Creating Custom Trainers:**

To create your own custom trainer (e.g., for different epoch counts):

1. Create a new Python file in the project root (e.g., `nnUNetTrainer_250epochs.py`)
2. Subclass `nnUNetTrainer` and override `num_epochs` and `max_num_epochs`
3. Use `--trainer nnUNetTrainer_250epochs` when training

**Example - 250 Epochs:**

```python
# nnUNetTrainer_250epochs.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_250epochs(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device='cuda'):
        super().__init__(plans, configuration, fold, dataset_json,
                        unpack_dataset, device)
        self.num_epochs = 250
        self.max_num_epochs = 250
```

Then train with:
```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --trainer nnUNetTrainer_250epochs \
    ...
```

### Multi-GPU Training

#### Parallel Fold Training

Train different folds on different GPUs:

```bash
# Terminal 1 - GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --fold_range 0-4

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --fold_range 5-9
```

#### Configuration File

The `train_config.json` file controls device settings and training behavior:

**Example configuration**:
```json
{
  "experiment": {
    "name": "liver_cancer_nnunet_with_test",
    "description": "3-class liver cancer segmentation with proper test split and nnUNet",
    "random_seed": 42
  },
  "data": {
    "images_dir": "data/images",
    "labels_dir": "data/labels",
    "num_classes": 4,
    "class_names": ["background", "metastatic", "hcc", "cho"],
    "input_channels": 3
  },
  "training": {
    "continue_on_error": false,
    "num_workers": 12
  },
  "device": {
    "use_cuda": true,
    "cuda_device": 0
  }
}
```

**For multi-GPU training** (optional):
```json
{
  "device": {
    "use_cuda": true,
    "cuda_device": 0,
    "available_gpus": [0, 1],
    "fold_gpu_mapping": {
      "0-4": 0,
      "5-9": 1
    }
  },
  "training": {
    "num_workers": 12,
    "continue_on_error": false
  }
}
```

Use with:
```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 104 \
    --config train_config.json
```

**Note**: The `max_epochs` parameter is recognized but not used by nnUNet (see "Custom Training Parameters" section).

### Custom Training Parameters

nnUNet v2 determines most parameters automatically. Most parameters cannot be overridden via command line.

**Why can't I set max epochs?**

nnUNet v2 does not support `--num_epochs` or `--max_epochs` flags. It uses adaptive training:
- Default: 1000 epochs maximum
- Early stopping: Automatic (patience ~50 epochs)
- This is the scientifically validated approach

**If you need custom epochs** (advanced):

Create a custom nnUNet trainer:
```python
# custom_trainer.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class CustomTrainer(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 500  # Custom epoch count
        self.initial_lr = 0.001  # Custom learning rate
```

Use with:
```bash
nnUNetv2_train 104 2d 0 -tr CustomTrainer
```

**Recommendation**: Use nnUNet's default adaptive training. It's optimized for medical image segmentation and will stop early if convergence is detected.

### Inference After Training

Once training completes:

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --input_dir nnUNet_raw/Dataset104_LiverCancerStratified/imagesTs \
    --output_dir predictions/dataset104 \
    --configuration 2d \
    --all_folds \
    --compute_metrics \
    --checkpoint checkpoint_best
```

See `inference.md` for complete inference documentation.

---

## Training Timeline

### Expected Duration

**Hardware**: NVIDIA RTX 3090 (24GB VRAM)

| Configuration | Per Fold | All 10 Folds | Both Configs |
|---------------|----------|--------------|--------------|
| 2D            | 2-4 hours| 20-40 hours  | -            |
| 3D_fullres    | 8-12 hours| 80-120 hours| -            |
| **Total**     | -        | -            | **100-160 hours** |

**Preprocessing**: 1-2 hours (one-time)

### Faster Training

**Reduce training time**:
1. Train 2D only (4x faster than 3D)
2. Use fewer folds (e.g., 5-fold instead of 10)
3. Use better GPU (A100: 2-3x faster than RTX 3090)
4. Parallel training (multiple GPUs)

**For testing/development**:
```bash
# Train single fold only
python scripts/segmentation/nnunet_training.py --dataset_id 104 --fold 0

# Estimated time: 2-4 hours (2D) or 8-12 hours (3D)
```

---

## Best Practices

### DO ✅

1. **Run verification before training**
   ```bash
   python scripts/preprocessing/verify_partitions.py --dataset_id 104 --dataset_name LiverCancerStratified
   ```

2. **Save training metadata** - Essential for publication

3. **Monitor training** - Check logs regularly for errors

4. **Use resume capability** - Don't restart from scratch if interrupted

5. **Document everything** - Future you will thank you

6. **Test on single fold first** - Verify pipeline before full training

### DON'T ❌

1. **Don't skip verification** - May lead to invalid results

2. **Don't delete metadata** - Needed for publication

3. **Don't modify splits after training starts** - Causes leakage

4. **Don't cherry-pick best fold** - Report mean ± std across all folds

5. **Don't use validation set for final metrics** - Must use held-out test set

6. **Don't mix training sessions** - Each session gets unique ID

---

## Quick Reference

### Command Cheat Sheet

```bash
# Standard workflow
python scripts/segmentation/nnunet_training.py --dataset_id 104 --dataset_name LiverCancerStratified

# Preprocessing only
python scripts/segmentation/nnunet_training.py --dataset_id 104 --plan_only

# Training only (skip preprocessing)
python scripts/segmentation/nnunet_training.py --dataset_id 104 --train_only

# Specific fold
python scripts/segmentation/nnunet_training.py --dataset_id 104 --fold 0

# Fold range
python scripts/segmentation/nnunet_training.py --dataset_id 104 --fold_range 0-4

# Resume interrupted training
python scripts/segmentation/nnunet_training.py --dataset_id 104 --resume

# Force restart
python scripts/segmentation/nnunet_training.py --dataset_id 104 --force_restart

# Custom seed
python scripts/segmentation/nnunet_training.py --dataset_id 104 --random_seed 12345

# Specific configurations
python scripts/segmentation/nnunet_training.py --dataset_id 104 --configurations 2d

# Verify after training
python scripts/preprocessing/verify_partitions.py --dataset_id 104 --dataset_name LiverCancerStratified
```

### File Locations

```bash
# Training metadata (per session)
training_metadata/Dataset104_LiverCancerStratified/<session_id>/

# nnUNet results (all models)
nnUNet_results/Dataset104_LiverCancerStratified/

# Preprocessed data
nnUNet_preprocessed/Dataset104_LiverCancerStratified/

# Raw dataset
nnUNet_raw/Dataset104_LiverCancerStratified/
```

### Log Files

```bash
# Main training log
nnunet_training.log

# Per-fold logs
training_metadata/Dataset104_*/*/training_2d_fold0.log

# Training summary
training_metadata/Dataset104_*/*/training_summary.json
training_metadata/Dataset104_*/*/training_summary.txt

# System info
training_metadata/Dataset104_*/*/system_info.json
training_metadata/Dataset104_*/*/software_versions.json
```

---

## References

### Related Documentation

- `preprocessing.md` - Dataset preparation and conversion
- `inference.md` - Test set evaluation and metrics
- `verification.md` - Data leakage verification procedures

### Scripts

- `scripts/segmentation/nnunet_training.py` - Main training script (use this!)
- `scripts/segmentation/nnunet_training.py` - Old version (deprecated, has bugs)
- `scripts/preprocessing/verify_partitions.py` - Verify preprocessing
- `scripts/preprocessing/nnunet_conversion_with_partitions.py` - Dataset conversion

### External Resources

- [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)
- [nnU-Net Paper](https://www.nature.com/articles/s41592-020-01008-z)
- [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation)

### Support

For issues with:
- **This training script**: Check this documentation
- **nnUNet itself**: See [nnUNet issues](https://github.com/MIC-DKFZ/nnUNet/issues)
- **CUDA/PyTorch**: Check [PyTorch forums](https://discuss.pytorch.org/)

---

**Last Updated**: January 2025
**Version**: 1.0
**Author**: Liver Cancer Segmentation Project

---

## Appendix: Training Hyperparameters

### nnUNet Default Hyperparameters

nnUNet automatically determines these based on your dataset:

**Network Architecture**:
- Encoder: 5-6 levels (depends on image size)
- Initial features: 32
- Max features: 320
- Kernel sizes: 3×3 (2D) or 3×3×3 (3D)
- Batch normalization: Instance normalization
- Activation: LeakyReLU

**Training**:
- Optimizer: SGD with Nesterov momentum (0.99)
- Initial learning rate: 0.01
- LR schedule: Polynomial decay (exponent 0.9)
- Weight decay: 3e-5
- Loss: Dice + CE (equal weights)
- Batch size: Auto-determined (typically 2-3)
- Patch size: Auto-determined from image size
- Epochs: 1000 (adaptive early stopping)
- Validation: Every epoch
- Checkpoint: Save best validation Dice

**Note on Epochs**: nnUNet uses adaptive training with 1000 epochs maximum and automatic early stopping. The `max_epochs` parameter in `train_config.json` is not used because nnUNet v2 does not support custom epoch limits via CLI. Training will stop early if validation performance plateaus (typically 50 epochs patience).

**Data Augmentation**:
- Rotation: ±30°
- Scaling: 0.7-1.4
- Elastic deformation: Yes
- Gaussian noise: σ=0.1
- Gaussian blur: σ=0.5-1.0
- Brightness: ±0.3
- Contrast: 0.75-1.25
- Gamma: 0.7-1.5

These are logged in `nnUNetPlans.json` for your specific dataset.

### Accessing Your Training Hyperparameters

```bash
# View plans file
cat nnUNet_preprocessed/Dataset104_LiverCancerStratified/nnUNetPlans.json | jq

# Extract key parameters
python -c "
import json
with open('nnUNet_preprocessed/Dataset104_LiverCancerStratified/nnUNetPlans.json') as f:
    plans = json.load(f)
    config = plans['configurations']['2d']
    print(f'Batch size: {config[\"batch_size\"]}')
    print(f'Patch size: {config[\"patch_size\"]}')
"
```
