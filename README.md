# Liver Cancer Segmentation & Classification - Study Replication Package

Supplementary code and procedures for our liver cancer study on 3-channel MRI.
The pipeline performs **multi-class segmentation** (4 classes: background, metastatic,
HCC, CHO) with nnU-Net v2, and **patch-based classification / diagnosis** with five
modern CNN/transformer backbones, evaluated with rigorous patient-level
cross-validation and held-out test partitions.

This repository contains everything needed to **replicate the study end-to-end**:
data preprocessing/conversion, segmentation training & inference, classification
training & inference, metrics, and external-image evaluation. The large imaging
data and trained weights are published in the FULIR / IRB repository
(<https://data.fulir.irb.hr/en/object/irb:896>) as two separate archives (see
[Data & weights](#3-data--weights)).

---

## Quickstart

> **The one rule that prevents every path problem: always work from the project root** — the folder
> that contains `scripts/`, `release_tools/`, and this `README.md`. Every command below is run from
> there. The scripts locate `data/`, `nnUNet_raw/`, `nnUNet_results/`, etc. relative to your current
> directory, so the project root must be your working directory.

```bash
# 1. Enter the project folder (the one containing scripts/ and release_tools/)
cd path/to/HandE-Liver3C                 # adjust to where you cloned / unzipped it

# 2. Create the environment and install dependencies (see Section 2 for details)
python -m venv .venv
# Windows:      .venv\Scripts\Activate.ps1
# Linux/macOS:  source .venv/bin/activate
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 3. Restore data + weights. The download URLs are already configured (hosted in the
#    FULIR / IRB repository, https://data.fulir.irb.hr/en/object/irb:896). The script
#    auto-detects the project root and extracts there:
#    data/, nnUNet_raw/, nnUNet_preprocessed/, nnUNet_results/, classification_results/
python release_tools/download_and_extract.py

# 4. Run any pipeline step — always from the project root:
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 100 --dataset_name Liver1 \
    --input_dir nnUNet_raw/Dataset100_Liver1/imagesTs \
    --output_dir predictions/Partition_1/complete \
    --configuration 2d --all_folds --trainer nnUNetTrainer_500epochs \
    --compute_metrics --labels_dir nnUNet_raw/Dataset100_Liver1/labelsTs
```

**If a script reports a missing `nnUNet_raw/`, `nnUNet_results/`, or `data/`, you are not in the project
root — `cd` back to it and re-run.** `download_and_extract.py` restores into the project root even if you
launch it from elsewhere (it finds the root from its own location); the pipeline scripts, however, must be
launched from the project root.

---

## Contents

| Path | What it is |
|------|------------|
| `scripts/` | The 18 pipeline scripts grouped by stage: `preprocessing/`, `segmentation/`, `classification/`, `external/`, `metrics/` |
| `train_config.json` | nnU-Net training config (device, workers, seed) - kept at project root |
| `classification_config.json` | Patch-classification config (patch size, batch, classes) - kept at project root |
| `nnUNetTrainer_500epochs.py` | Custom nnU-Net trainer (500 epochs) - kept at project root so nnU-Net's training subprocess can import it |
| `documentation/` | Detailed procedure guides (preprocessing, training, inference, metrics) |
| `metadata/` | CV splits, partition definitions, and best-fold selections (small, version-controlled) |
| `release_tools/` | Scripts to package and to download/extract the data & weights archives |
| `requirements.txt` | Pinned dependencies |

> **Authoritative commands live in this README.** The guides under `documentation/`
> contain additional background and troubleshooting; where an older script name appears
> there, the mapping is: `nnunet_training_rigorous.py` -> **`scripts/segmentation/nnunet_training.py`**,
> `nnunet_conversion_with_stratified_test.py` -> **`scripts/preprocessing/nnunet_conversion_with_partitions.py`**,
> verification scripts -> **`scripts/preprocessing/verify_partitions.py`**.

> **Run every script from the project root** (the directory holding `data/`, `nnUNet_raw/`,
> `nnUNet_results/`, `scripts/`, …), e.g. `python scripts/segmentation/nnunet_training.py …`. The
> scripts resolve project directories relative to the current working directory, so the project root
> must be your working directory regardless of which subfolder a script lives in.

---

## 1. Datasets and cross-validation

The study uses **5 test partitions**, each materialised as an independent nnU-Net dataset
with its own held-out test set and a 10-fold patient-level CV over the training portion:

| Dataset ID | Name | Partition |
|------------|------|-----------|
| 100 | Liver1 | 1 |
| 101 | Liver2 | 2 |
| 102 | Liver3 | 3 |
| 103 | Liver4 | 4 |
| 104 | Liver5 | 5 |

Classes: `0=background, 1=metastatic, 2=hcc, 3=cho`. Input images have 3 channels.
The exact partition definitions and CV splits are in
`metadata/test_partitions.json` and `metadata/splits/DatasetNNN_LiverX/splits_final.json`.

---

## 2. Installation

```bash
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\Activate.ps1

# Install the pinned CUDA 12.6 PyTorch build first, then the rest:
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

For a different CUDA version or CPU-only, install the matching torch build from
<https://pytorch.org/get-started/locally/> and ignore the `+cu126` pins.
A CUDA GPU with >=11 GB VRAM is recommended for training.

---

## 3. Data & weights

Both are published in the **FULIR / IRB repository** and restored into the correct project
locations by a helper script. The script extracts into the **project root** — which it auto-detects from
its own location, so it lands correctly even if you launch it from another directory — recreating
`data/`, `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`, and `classification_results/` exactly
where the pipeline scripts look for them.

- **Repository / DOI landing page:** <https://data.fulir.irb.hr/en/object/irb:896>
- **Direct dataset download:** <https://data.fulir.irb.hr/data/HandE-Liver3C/HandE-Liver3C_data.zip>
- **Direct weights download:** <https://data.fulir.irb.hr/data/HandE-Liver3C/HandE-Liver3C_weights.zip>

The download URLs are already configured in `release_tools/download_and_extract.py`, so no editing is
required:

```bash
# Restore everything into the project root (downloads both archives from the repository):
python release_tools/download_and_extract.py

# Or use already-downloaded zips:
python release_tools/download_and_extract.py \
    --data-zip ./HandE-Liver3C_data.zip --weights-zip ./HandE-Liver3C_weights.zip

# Data only (skip the large weights archive):
python release_tools/download_and_extract.py --skip-weights
```

| Archive | Restores to | Notes |
|---------|-------------|-------|
| `HandE-Liver3C_data.zip` (~2.4 GB) | `data/`, `nnUNet_raw/` | Raw `.npy` images+labels, external JPGs, and the converted nnU-Net datasets |
| `HandE-Liver3C_weights.zip` (~226 GB) | `nnUNet_results/`, `classification_results/` | **All folds** of every model (each fold's `checkpoint_best.pth`) |

**Full ensemble reproducible.** The weights archive ships every fold of every model, so the
documented `--all_folds` ensemble inference (Sections 4-5) reproduces the paper exactly. For
single-fold use, the best fold per model is recorded in `metadata/best_folds.json`
(segmentation) and `metadata/classification_best_folds.json` (classification). Maintainers
re-hosting the release can build a much smaller best-fold-only archive by running
`release_tools/package_weights.py` **without** `--all-folds` (see the note below).

`nnUNet_raw/` can also be regenerated from `data/` (Section 4), so a code-only user may
skip it.

> **Maintainer note — rebuilding the archives.** End users do **not** need this. The two
> hosted zips are produced from a full project checkout (one that still holds the heavy data
> and weights) by `release_tools/package_data.py` and `release_tools/package_weights.py`, which
> preserve archive paths relative to the project root so `download_and_extract.py` restores them
> to the expected locations. Each prints the archive's SHA256 — paste it into `DATA_SHA256` /
> `WEIGHTS_SHA256` in `download_and_extract.py` before re-hosting. Pass `--all-folds` to
> `package_weights.py` for the full-ensemble release, or omit it for a best-fold-only archive.

---

## 4. Segmentation pipeline (nnU-Net)

**Step 1 - Convert raw data into the 5 nnU-Net datasets** (creates `nnUNet_raw/Dataset100..104`
with stratified test splits and 10-fold CV):

```bash
python scripts/preprocessing/nnunet_conversion_with_partitions.py --partitions_file data/test_partitions.json
python scripts/preprocessing/verify_partitions.py            # check no patient leakage across train/val/test
```

**Step 2 - Train** (per dataset; 500-epoch trainer, all 10 folds). nnU-Net planning &
preprocessing run automatically on first invocation:

```bash
python scripts/segmentation/nnunet_training.py \
    --dataset_id 100 --dataset_name Liver1 \
    --config train_config.json \
    --trainer nnUNetTrainer_500epochs \
    --fold_range 0-9 --random_seed 42
```

Repeat for `--dataset_id 101 --dataset_name Liver2`, ... `104 / Liver5`. Use `--fold N` for a
single fold, `--plan_only` / `--train_only` to split preprocessing from training. See
`documentation/segmentation_training.md` (and `documentation/WINDOWS_BACKGROUND_TRAINING.md`
for running detached on Windows).

**Step 3 - Inference + metrics.** Full ensemble across all 10 folds:

```bash
python scripts/segmentation/nnunet_inference.py \
    --dataset_id 100 --dataset_name Liver1 \
    --input_dir nnUNet_raw/Dataset100_Liver1/imagesTs \
    --output_dir predictions/Partition_1/complete \
    --configuration 2d --all_folds \
    --trainer nnUNetTrainer_500epochs \
    --compute_metrics --labels_dir nnUNet_raw/Dataset100_Liver1/labelsTs --visualize
```

The weights archive ships all 10 folds, so `--all_folds` works directly. For a quick
single-fold run, replace it with `--fold N` using the best fold from `metadata/best_folds.json`
(e.g. Dataset100 -> fold 0).

Standalone metric computation on a predictions/labels pair:

```bash
python scripts/metrics/compute_test_metrics.py \
    --predictions predictions/Partition_1/complete/ensemble \
    --labels nnUNet_raw/Dataset100_Liver1/labelsTs --num_classes 4
```

---

## 5. Classification pipeline (patch-based diagnosis)

Extracts patches, classifies each with a `timm` backbone, then majority-votes per image.
Backbones: `convnextv2`, `efficientnetv2`, `swinv2`, `maxvit`, `densenet`.

**Train** (all five models, all folds, for one dataset):

```bash
python scripts/classification/classification_training.py \
    --dataset_id 100 --dataset_name Liver1 \
    --config classification_config.json \
    --all --fold_range 0-9 --pretrained --random_seed 42
```

Use `--model convnextv2` to train a single backbone. Repeat per dataset 100-104.
See `documentation/classification_training.md`.

**Inference + metrics** (per model; ensemble across all folds):

```bash
python scripts/classification/classification_inference.py \
    --dataset_id 100 --model convnextv2 \
    --all_folds --compute_metrics --visualize --generate_summary
```

All folds are shipped, so `--all_folds` works directly. For a single-fold run, use `--fold N`
with the best fold from `metadata/classification_best_folds.json`.

**Aggregate & majority-vote across the ensemble of datasets:**

```bash
python scripts/metrics/aggregate_classification_metrics.py        # per-dataset metric workbooks
python scripts/metrics/majority_vote_analysis.py \
    --predictions_dir predictions_classification \
    --output_dir predictions_classification/majority_vote
```

---

## 6. External-image evaluation (optional)

Evaluate the trained ensembles on external JPG images:

```bash
python scripts/external/rename_external_images.py           # standardise filenames (CHOL/COLON/HCC -> cho/hcc/metastatic)
python scripts/external/downsample_external_images.py       # match training resolution
python scripts/external/external_images_inference.py        # run nnU-Net + 5 classification ensembles
```

---

## 7. Script reference

| Script | Role |
|--------|------|
| `scripts/preprocessing/nnunet_conversion_with_partitions.py` | Build the 5 nnU-Net datasets + CV splits from `data/` |
| `scripts/preprocessing/verify_partitions.py` | Verify no patient leakage / stratification across folds |
| `scripts/segmentation/nnunet_training.py` | Rigorous, reproducible nnU-Net training (custom CV, metadata logging) |
| `nnUNetTrainer_500epochs.py` | 500-epoch trainer override |
| `scripts/segmentation/nnunet_inference.py` | Test-set inference (per-fold / ensemble), metrics, visualizations |
| `scripts/classification/nnunet_patch_classification.py` | Convert nnU-Net pixel predictions -> patch-level diagnosis |
| `scripts/classification/classification_training.py` | Patch-based classification training (timm backbones) |
| `scripts/classification/classification_inference.py` | Patch classification inference + majority-vote diagnosis |
| `scripts/external/external_images_inference.py` | Run all trained ensembles on external images |
| `scripts/external/downsample_external_images.py`, `scripts/external/rename_external_images.py` | External-image preprocessing |
| `scripts/metrics/compute_test_metrics.py` | Segmentation metrics (Dice / Jaccard / F1) |
| `scripts/metrics/compute_diagnosis_metrics.py`, `scripts/metrics/compute_patch_metrics.py` | Diagnosis- and patch-level metrics |
| `scripts/metrics/generate_metrics_summary.py`, `scripts/metrics/aggregate_classification_metrics.py` | Aggregate metrics (mean +/- std, Excel) |
| `scripts/metrics/find_best_folds.py` | Select best segmentation fold per dataset -> `best_folds.json` |
| `scripts/metrics/export_best_fold_metrics.py` | Export best-fold metrics to Excel |
| `scripts/metrics/majority_vote_analysis.py` | Image-level majority vote across the dataset ensemble |
| `release_tools/package_data.py` | *(maintainer only)* Rebuild `HandE-Liver3C_data.zip` from a full checkout for re-hosting |
| `release_tools/package_weights.py` | *(maintainer only)* Rebuild `HandE-Liver3C_weights.zip` (all folds, or best-fold without `--all-folds`) for re-hosting |
| `release_tools/download_and_extract.py` | Fetch & unpack the hosted archives (what end users run) |

---

## 8. Reproducibility notes

- All training scripts fix `--random_seed 42` and log system/software versions and the exact
  CV splits. The canonical splits and per-dataset preprocessing artifacts are version-controlled
  under `metadata/splits/` and `metadata/preprocessed/`. Running training also writes a full
  per-run record to `training_metadata/` on your machine.
- Classification inference normalizes patches using each dataset's
  `nnUNet_preprocessed/<dataset>/dataset_fingerprint.json` (restored by the data archive). Without
  it the code falls back to mean=0.5/std=0.5, which will not reproduce the reported numbers.
- Fully bit-identical results require the same GPU model, CUDA, PyTorch, and nnU-Net versions;
  metrics are otherwise expected within +/-0.5% Dice. See `documentation/segmentation_training.md`.

## Data usage & ethics

The imaging data is de-identified medical data provided for research replication. By
downloading the data archive you agree to use it for non-commercial research only and not to
attempt re-identification. The MIT `LICENSE` covers the **code**; the data and weights are
released under the terms described here. If you use this work, please cite the accompanying
paper.
