# nnUNet Dataset Preprocessing Guide

**Scientific Medical Image Segmentation with Data Integrity**

This document describes the complete preprocessing pipeline for liver cancer segmentation, ensuring scientific rigor, reproducibility, and prevention of data leakage.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Train/Test Split Strategy](#traintestsplit-strategy)
4. [Stratified Cross-Validation](#stratified-cross-validation)
5. [Data Leakage Prevention](#data-leakage-prevention)
6. [nnUNet Conversion](#nnunet-conversion)
7. [Verification Procedures](#verification-procedures)
8. [For Publication](#for-publication)

---

## Overview

### Scientific Requirements

For publication in top-tier ML/medical imaging venues (MICCAI, Medical Image Analysis), the preprocessing pipeline must ensure:

1. **Strict train/test separation** - Test set never seen during training/validation
2. **Patient-level splitting** - Same patient not in both train and test
3. **Stratified cross-validation** - Balanced class distribution across folds
4. **Reproducibility** - Fixed random seeds, documented procedures
5. **No data leakage** - Preprocessing uses only training data

### Pipeline Overview

```
Raw Data (.npy files)
    ↓
Dataset Conversion (nnunet_conversion_with_partitions.py)
    ↓
├── Test Set (18 patients, 36 cases) → Hold out completely
└── Train/Val Set (71 patients, 134 cases)
        ↓
    10-Fold Stratified CV → Balanced class distribution
        ↓
    nnUNet Format (PNG files) → Ready for preprocessing
        ↓
    nnUNet Preprocessing → Statistics from training data only
        ↓
    Training → Using custom splits
```

---

## Dataset Preparation

### Input Data Structure

Your raw data should be organized as:

```
data/
├── images/           # 3-channel MRI images
│   ├── cho_10.npy
│   ├── cho_10a.npy   # Second image from same patient
│   ├── hcc_15.npy
│   ├── metastatic_20.npy
│   └── ...
└── labels/           # One-hot encoded labels
    ├── cho_10.npy    # Shape: (H, W, 4) - [background, metastatic, HCC, CHO]
    ├── cho_10a.npy
    └── ...
```

**File Format**:
- Images: `.npy` files, shape `(H, W, 3)`, 3-channel MRI
- Labels: `.npy` files, shape `(H, W, 4)`, one-hot encoded
  - Channel 0: Background
  - Channel 1: Metastatic tumor
  - Channel 2: HCC (Hepatocellular Carcinoma)
  - Channel 3: CHO (Cholangiocarcinoma)

**Patient ID Convention**:
- Files ending with 'a' (e.g., `cho_10a.npy`) belong to same patient as base name
- Patient ID: `extract_patient_id('cho_10a')` → `'cho_10'`

---

## Train/Test Split Strategy

### Methodology

**Test Set Selection Criteria**:
- Only patients with **exactly 2 images** are eligible for test set
- Ensures test set has consistent data quality
- 6 patients per cancer type (18 total patients, 36 cases)
- Balanced across classes: 6 CHO, 6 HCC, 6 Metastatic

**Train/Val Set**:
- All remaining patients (1+ images per patient)
- Used for 10-fold cross-validation
- Never mixed with test set

### Scientific Rationale

**Why only patients with 2 images for test set?**
- Ensures test set quality (complete imaging protocols)
- Allows paired analysis if needed
- Documents selection bias for transparency

**Why 6 patients per class?**
- Total: 18 patients × 2 images = 36 test cases
- Provides sufficient test set size (~20% of data)
- Maintains class balance

**Why patient-level splitting?**
- Prevents data leakage (same patient in train/test)
- Reflects clinical scenario (new patient prediction)
- Standard practice in medical imaging

---

## Stratified Cross-Validation

### Implementation

**10-Fold Stratified K-Fold**:
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(patients, labels):
    # Ensures balanced class distribution in each fold
```

### Class Distribution

**Before Stratification (Random Split)**:
```
Fold 0: 62% CHO, 25% HCC, 13% Metastatic  ← Imbalanced!
Fold 2: 14% CHO, 29% HCC, 57% Metastatic  ← Imbalanced!
```

**After Stratification**:
```
All folds: ~32% CHO, ~34% HCC, ~34% Metastatic  ← Balanced!
```

### Benefits

1. **Reduced variance** in cross-validation estimates
2. **Fair training** - each fold sees all classes proportionally
3. **Reliable model selection** - consistent validation performance
4. **Standard practice** - required by MICCAI/NeurIPS

---

## Data Leakage Prevention

### Critical Points

**❌ Common Leakage Sources**:
1. Computing normalization statistics on train + test
2. Creating CV splits after preprocessing
3. Using test set for hyperparameter tuning
4. Evaluating on validation set multiple times without correction

**✅ Our Prevention Measures**:
1. Test set separated BEFORE any preprocessing
2. CV splits created BEFORE nnUNet preprocessing
3. Preprocessing only sees training data
4. Test set used ONCE for final evaluation

### Verification

After dataset conversion, verify no leakage:

```bash
# Comprehensive verification
python scripts/preprocessing/verify_partitions.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified
```

**Expected output**:
```
✅ Fingerprint analyzed 134 cases (train only)
✅ No preprocessed test set found
✅ CV splits contain only training cases
✅ Plans file doesn't reference test set
```

**Red flags**:
```
🔴 LEAKAGE: Fingerprint analyzed 170 cases (train + test!)
🔴 LEAKAGE: Test set was preprocessed!
🔴 FOUND TEST CASES IN CV SPLITS
```

---

## nnUNet Conversion

### Step-by-Step Procedure

#### 1. Prepare Environment

```bash
cd /path/to/project
source .venv/bin/activate  # Or your virtual environment
```

#### 2. Run Conversion Script

```bash
python scripts/preprocessing/nnunet_conversion_with_partitions.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified \
    --images_dir data/images \
    --labels_dir data/labels \
    --test_patients_per_class 6 \
    --n_folds 10 \
    --random_seed 42
```

**Parameters**:
- `--dataset_id`: Unique 3-digit ID (e.g., 104)
- `--dataset_name`: Descriptive name for this dataset
- `--test_patients_per_class`: Patients per class for test set (default: 6)
- `--n_folds`: Number of CV folds (default: 10)
- `--random_seed`: For reproducibility (default: 42)

#### 3. Verify Output Structure

```bash
ls -R nnUNet_raw/Dataset104_LiverCancerStratified/
```

**Expected structure**:
```
nnUNet_raw/Dataset104_LiverCancerStratified/
├── imagesTr/              # Training images
│   ├── cho_10_0000.png    # Channel 0
│   ├── cho_10_0001.png    # Channel 1
│   ├── cho_10_0002.png    # Channel 2
│   └── ...
├── labelsTr/              # Training labels
│   ├── cho_10.png         # Single-channel, class indices
│   └── ...
├── imagesTs/              # Test images (held out)
│   └── ...
├── labelsTs/              # Test labels (for evaluation)
│   └── ...
├── dataset.json           # nnUNet dataset configuration
├── splits_final.json      # 10-fold CV splits (STRATIFIED)
└── test_set_info.json     # Test set metadata
```

#### 4. Verify Conversion

```bash
# Check file counts
echo "Training images: $(ls nnUNet_raw/Dataset104_*/imagesTr/*_0000.png | wc -l)"
echo "Test images: $(ls nnUNet_raw/Dataset104_*/imagesTs/*_0000.png | wc -l)"

# Check splits
python -c "
import json
with open('nnUNet_raw/Dataset104_LiverCancerStratified/splits_final.json') as f:
    splits = json.load(f)
print(f'Number of folds: {len(splits)}')
print(f'Fold 0 train: {len(splits[0][\"train\"])} cases')
print(f'Fold 0 val: {len(splits[0][\"val\"])} cases')
"
```

---

## Verification Procedures

### Checklist

Before starting training, verify:

- [ ] **Dataset conversion completed without errors**
  ```bash
  # Check log output for any warnings
  ```

- [ ] **File counts match expectations**
  ```bash
  # Training: 134 cases × 3 channels = 402 image files
  # Test: 36 cases × 3 channels = 108 image files
  ```

- [ ] **Splits are stratified**
  ```bash
  python scripts/preprocessing/verify_partitions.py \
      --dataset_id 104 \
      --dataset_name LiverCancerStratified
  ```
  Expected: Balanced class distribution per fold

- [ ] **No patient leakage (train/test)**
  ```bash
  python scripts/preprocessing/verify_partitions.py \
      --dataset_id 104 \
      --dataset_name LiverCancerStratified
  ```
  Expected: Zero overlap between train and test patients

- [ ] **No data leakage in preprocessing** (Run after nnUNet preprocessing)
  ```bash
  python scripts/preprocessing/verify_partitions.py \
      --dataset_id 104 \
      --dataset_name LiverCancerStratified
  ```
  Expected: All checks pass

### Expected Outputs

#### verify_partitions.py

```
✅ NO PATIENT LEAKAGE FOUND IN ANY CROSS-VALIDATION FOLD!
✅ All patients appear in exactly 9 training folds and 1 validation fold
Total unique patients in CV: 71
```

#### verify_partitions.py

```
✅ SUCCESS: No patient overlap found!
Test patients: 18 unique patients
Train/Val patients: 71 unique patients
```

#### verify_partitions.py

```
CHECK 1: ✅ PASS: Only training cases were analyzed
CHECK 2: ✅ GOOD: No preprocessed test set found
CHECK 3: ✅ GOOD: Preprocessed splits match raw dataset splits
CHECK 4: ✅ Plans file contains no test set references

FINAL VERDICT: ✅ ALL CHECKS PASSED
```

---

## For Publication

### Methods Section Template

```markdown
## Data Preprocessing

### Dataset Preparation

Our dataset consisted of 170 liver MRI scans from 89 patients with three
types of liver cancer: metastatic tumors, hepatocellular carcinoma (HCC),
and cholangiocarcinoma (CHO). Each scan was a 3-channel MRI image with
corresponding segmentation masks annotated by expert radiologists.

### Train/Test Split

To ensure unbiased evaluation, we created a held-out test set before any
preprocessing or model development. We selected 18 patients (6 per cancer
type) with exactly 2 images each, resulting in 36 test cases. This test
set was isolated and used exclusively for final evaluation. The remaining
71 patients (134 cases) were used for training and validation.

Patient-level splitting ensured no data leakage, as all images from the
same patient were assigned to the same set (train/validation or test).
This reflects the clinical scenario of predicting on new, unseen patients.

### Cross-Validation Strategy

We performed 10-fold stratified cross-validation on the training set.
Stratification ensured balanced class distribution across all folds
(~32% CHO, ~34% HCC, ~34% Metastatic in each validation set), reducing
variance in performance estimates and enabling reliable model selection.

All cross-validation splits were created at the patient level, maintaining
strict separation between training and validation patients within each fold.

### Preprocessing Pipeline

Dataset conversion to nnUNet format was performed using our custom script
(available in supplementary materials). Images were converted from numpy
arrays to PNG format, and labels were converted from one-hot encoding to
class indices.

nnUNet v2.x preprocessing was applied exclusively to the training data.
We verified that no test set information leaked into preprocessing by
confirming that:
1. Dataset statistics were computed from 134 training cases only
2. No preprocessed test set files were created
3. Cross-validation splits contained only training cases
4. nnUNet plans referenced only training data

### Reproducibility

All data splits were created with a fixed random seed (42) to ensure
reproducibility. Dataset preparation scripts and cross-validation splits
are provided in supplementary materials.
```

### Supplementary Materials

Include in your paper's supplementary materials:
1. `scripts/preprocessing/nnunet_conversion_with_partitions.py` - Conversion script
2. `splits_final.json` - Exact CV splits used
3. `test_set_info.json` - Test set composition
4. `verify_*.py` - Verification scripts
5. Complete preprocessing logs

---

## Troubleshooting

### Issue: "Dataset not found"

**Problem**: Raw data directory doesn't exist or is incorrectly specified

**Solution**:
```bash
# Check data directory structure
ls -la data/images/ | head
ls -la data/labels/ | head

# Run conversion with explicit paths
python scripts/preprocessing/nnunet_conversion_with_partitions.py \
    --images_dir /absolute/path/to/images \
    --labels_dir /absolute/path/to/labels
```

### Issue: "Not enough test-eligible patients"

**Problem**: Not enough patients with exactly 2 images

**Solution**:
```bash
# Check patient image counts
python -c "
from pathlib import Path
from collections import defaultdict

images_dir = Path('data/images')
patient_counts = defaultdict(int)

for f in images_dir.glob('*.npy'):
    patient_id = f.stem[:-1] if f.stem.endswith('a') else f.stem
    patient_counts[patient_id] += 1

for class_name in ['cho', 'hcc', 'metastatic']:
    patients_2img = [p for p, c in patient_counts.items()
                     if p.startswith(class_name) and c == 2]
    print(f'{class_name}: {len(patients_2img)} patients with 2 images')
"

# Reduce test_patients_per_class if needed
python scripts/preprocessing/nnunet_conversion_with_partitions.py \
    --test_patients_per_class 4  # Reduced from 6
```

### Issue: "Class imbalance in folds"

**Problem**: Non-stratified splits were created

**Solution**: Use `scripts/preprocessing/nnunet_conversion_with_partitions.py` (not the old version)

### Issue: "Test set leakage detected"

**Problem**: Preprocessing used test set data

**Solution**:
```bash
# Delete preprocessed data
rm -rf nnUNet_preprocessed/Dataset104_*

# Ensure splits exist before preprocessing
ls nnUNet_raw/Dataset104_*/splits_final.json

# Re-run preprocessing (training script will do this)
python scripts/segmentation/nnunet_training.py --dataset_id 104
```

---

## Quick Reference

### Commands Summary

```bash
# 1. Convert dataset (with stratification)
python scripts/preprocessing/nnunet_conversion_with_partitions.py \
    --dataset_id 104 \
    --dataset_name LiverCancerStratified

# 2. Verify no patient leakage (train/test)
python scripts/preprocessing/verify_partitions.py --dataset_id 104 --dataset_name LiverCancerStratified

# 3. Verify stratified CV
python scripts/preprocessing/verify_partitions.py --dataset_id 104 --dataset_name LiverCancerStratified

# 4. Train (includes preprocessing)
python scripts/segmentation/nnunet_training.py --dataset_id 104 --dataset_name LiverCancerStratified

# 5. Verify no preprocessing leakage (after training starts)
python scripts/preprocessing/verify_partitions.py --dataset_id 104 --dataset_name LiverCancerStratified
```

### File Locations

```bash
Raw data:          data/images/, data/labels/
nnUNet raw:        nnUNet_raw/Dataset104_LiverCancerStratified/
nnUNet preproc:    nnUNet_preprocessed/Dataset104_LiverCancerStratified/
nnUNet results:    nnUNet_results/Dataset104_LiverCancerStratified/
Training metadata: training_metadata/Dataset104_LiverCancerStratified/
```

---

## Best Practices

### DO ✅

1. **Always verify** - Run all verification scripts before training
2. **Document everything** - Save all metadata and logs
3. **Use stratification** - Ensures balanced cross-validation
4. **Patient-level splits** - Prevents data leakage
5. **Fixed random seeds** - Enables reproducibility
6. **Test set last** - Only use once for final evaluation

### DON'T ❌

1. **Don't skip verification** - May lead to invalid results
2. **Don't use test set during development** - Will inflate performance
3. **Don't modify splits after preprocessing** - Causes leakage
4. **Don't mix patient images** - Same patient must stay together
5. **Don't report validation metrics** - Must report test set performance
6. **Don't cherry-pick results** - Report all trained configurations

---

## References

### Scripts

- `scripts/preprocessing/nnunet_conversion_with_partitions.py` - Dataset conversion with stratification
- `scripts/preprocessing/verify_partitions.py` - Verify train/test patient separation
- `scripts/preprocessing/verify_partitions.py` - Verify CV fold patient separation
- `scripts/preprocessing/verify_partitions.py` - Verify preprocessing data separation
- `check_nnunet_leakage.py` - Quick leakage check

### Documentation

- `training.md` - Training procedure and monitoring
- `inference.md` - Test set evaluation and metrics

### External Resources

- [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [Medical Image Analysis Best Practices](https://link.springer.com/journal/11548)
- [MICCAI Submission Guidelines](https://conferences.miccai.org/)

---

**Last Updated**: January 2025
**Version**: 1.0
**Author**: Liver Cancer Segmentation Project
