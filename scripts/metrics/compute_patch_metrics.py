#!/usr/bin/env python3
"""
Compute per-patch classification metrics from existing predictions.
Reads predictions from predictions_classification/ and computes patch-level accuracy.

Usage:
    # All datasets and models
    python compute_patch_metrics.py --all

    # Specific dataset and model
    python compute_patch_metrics.py --dataset_id 104 --model convnextv2

    # All models for one dataset
    python compute_patch_metrics.py --dataset_id 104 --all_models
"""

import os
import json
import argparse
import csv
import sys
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Dataset ID to name mapping
DATASETS = {
    100: 'Liver1',
    101: 'Liver2',
    102: 'Liver3',
    103: 'Liver4',
    104: 'Liver5'
}

MODELS = ['convnextv2', 'efficientnetv2', 'swinv2', 'maxvit', 'densenet']

CLASS_NAMES = ['background', 'metastatic', 'hcc', 'cho']


def determine_patch_true_class(label: np.ndarray, x: int, y: int, patch_size: int,
                                min_cancer_threshold: float = 0.10) -> Optional[int]:
    """
    Determine ground truth class for a patch based on label region.

    Uses same logic as training:
    - 0% cancer pixels → background (0)
    - ≥10% of a tumor class → that tumor class (1, 2, or 3)
    - >0% but <10% → borderline (return None, skip this patch)

    Args:
        label: Full label image
        x, y: Top-left corner of patch
        patch_size: Patch size
        min_cancer_threshold: Minimum cancer percentage to classify as tumor

    Returns:
        Class index (0-3) or None if borderline patch
    """
    h, w = label.shape[:2]

    # Get actual patch region (may be smaller at edges)
    y_end = min(y + patch_size, h)
    x_end = min(x + patch_size, w)
    patch_label = label[y:y_end, x:x_end]

    total_pixels = patch_label.size
    if total_pixels == 0:
        return 0  # Empty patch = background

    # Count pixels per class
    counts = np.bincount(patch_label.flatten(), minlength=4)

    # Calculate tumor percentages
    tumor_counts = counts[1:]  # [metastatic, hcc, cho]
    total_tumor = tumor_counts.sum()
    tumor_percentage = total_tumor / total_pixels

    # If no tumor pixels, it's background
    if total_tumor == 0:
        return 0

    # If tumor percentage is below threshold, it's borderline - skip
    if tumor_percentage < min_cancer_threshold:
        return None  # Borderline patch

    # Otherwise, return the dominant tumor class
    dominant_tumor = np.argmax(tumor_counts) + 1  # +1 because we excluded background
    return dominant_tumor


def compute_patch_metrics_for_fold(predictions_dir: Path, labels_dir: Path,
                                   min_cancer_threshold: float = 0.10) -> Dict:
    """
    Compute patch-level metrics for a single fold or ensemble.

    Args:
        predictions_dir: Directory containing {case_id}_predictions.json files
        labels_dir: Directory containing ground truth labels

    Returns:
        Dictionary with patch-level metrics
    """
    # Find all prediction files
    pred_files = sorted(predictions_dir.glob("*_predictions.json"))

    if not pred_files:
        return None

    all_true = []
    all_pred = []
    skipped_borderline = 0
    total_patches = 0

    for pred_file in pred_files:
        case_id = pred_file.stem.replace('_predictions', '')

        # Skip diagnosis_predictions.json
        if case_id == 'diagnosis':
            continue

        # Load predictions
        with open(pred_file) as f:
            pred_data = json.load(f)

        patch_size = pred_data['patch_size']

        # Load ground truth label
        label_path = labels_dir / f"{case_id}.png"
        if not label_path.exists():
            print(f"Warning: Label not found for {case_id}, skipping")
            continue

        label = np.array(Image.open(label_path))

        # Process each patch
        for patch in pred_data['patches']:
            x, y = patch['x'], patch['y']
            pred_class = patch['prediction']

            # Determine true class for this patch
            true_class = determine_patch_true_class(
                label, x, y, patch_size, min_cancer_threshold
            )

            total_patches += 1

            if true_class is None:
                skipped_borderline += 1
                continue

            all_true.append(true_class)
            all_pred.append(pred_class)

    if len(all_true) == 0:
        return None

    # Compute metrics
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    metrics = {
        'total_patches': total_patches,
        'evaluated_patches': len(all_true),
        'skipped_borderline': skipped_borderline,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    metrics['confusion_matrix'] = cm.tolist()

    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)

    metrics['per_class'] = {}
    for i, class_name in enumerate(CLASS_NAMES):
        support = int(np.sum(y_true == i))
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': support
        }

    # Macro/weighted averages
    metrics['macro_precision'] = float(np.mean(precision))
    metrics['macro_recall'] = float(np.mean(recall))
    metrics['macro_f1'] = float(np.mean(f1))
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, labels=[0, 1, 2, 3], zero_division=0
    )

    return metrics


def save_patch_metrics_report(output_dir: Path, metrics: Dict, fold_name: str, suffix: str = ""):
    """Save patch metrics report to JSON and TXT files."""
    # Save JSON
    json_file = output_dir / f"patch_metrics_{fold_name}{suffix}.json"
    metrics_json = metrics.copy()
    with open(json_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    # Save text report
    txt_file = output_dir / f"patch_metrics_{fold_name}{suffix}.txt"
    with open(txt_file, 'w') as f:
        f.write(f"PATCH-LEVEL CLASSIFICATION REPORT - {fold_name}\n")
        f.write("="*80 + "\n")
        f.write(f"Total patches: {metrics['total_patches']}\n")
        f.write(f"Evaluated patches: {metrics['evaluated_patches']}\n")
        f.write(f"Skipped (borderline): {metrics['skipped_borderline']}\n")
        f.write("="*80 + "\n\n")

        f.write("OVERALL METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:                {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy:       {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa:           {metrics['kappa']:.4f}\n")
        f.write(f"Matthews Correlation:    {metrics['mcc']:.4f}\n")
        f.write(f"Macro F1:                {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1:             {metrics['weighted_f1']:.4f}\n\n")

        f.write("CONFUSION MATRIX:\n")
        f.write("-"*80 + "\n")
        f.write("Rows: True | Columns: Predicted\n\n")

        cm = np.array(metrics['confusion_matrix'])
        f.write(f"{'':15s}")
        for name in CLASS_NAMES:
            f.write(f"{name:>12s}")
        f.write("\n" + "-"*63 + "\n")

        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{name:15s}")
            for j in range(4):
                f.write(f"{cm[i, j]:12d}")
            f.write("\n")
        f.write("\n")

        f.write("PER-CLASS METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}\n")
        f.write("-"*63 + "\n")

        for class_name in CLASS_NAMES:
            m = metrics['per_class'][class_name]
            f.write(f"{class_name:<15s} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                   f"{m['f1']:>10.4f} {m['support']:>10d}\n")

        f.write("\n" + "="*80 + "\n")


def create_patch_metrics_summary(output_dir: Path, fold_metrics: Dict[str, Dict], suffix: str = ""):
    """Create summary CSV comparing patch metrics across folds."""
    summary_file = output_dir / f"patch_metrics_summary{suffix}.csv"

    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Fold', 'Total_Patches', 'Evaluated_Patches', 'Skipped_Borderline',
                  'Accuracy', 'Balanced_Accuracy', 'Kappa', 'MCC', 'Macro_F1', 'Weighted_F1']
        for cls in CLASS_NAMES:
            header.extend([f'{cls}_Precision', f'{cls}_Recall', f'{cls}_F1', f'{cls}_Support'])
        writer.writerow(header)

        # Write fold results
        fold_names = sorted([k for k in fold_metrics.keys() if k.startswith('fold_')],
                           key=lambda x: int(x.split('_')[1]))
        if 'ensemble' in fold_metrics:
            fold_names.append('ensemble')

        for fold_name in fold_names:
            m = fold_metrics[fold_name]
            if m is None:
                continue

            row = [
                fold_name,
                m['total_patches'],
                m['evaluated_patches'],
                m['skipped_borderline'],
                f"{m['accuracy']:.4f}",
                f"{m['balanced_accuracy']:.4f}",
                f"{m['kappa']:.4f}",
                f"{m['mcc']:.4f}",
                f"{m['macro_f1']:.4f}",
                f"{m['weighted_f1']:.4f}",
            ]
            for cls in CLASS_NAMES:
                pc = m['per_class'][cls]
                row.extend([f"{pc['precision']:.4f}", f"{pc['recall']:.4f}",
                           f"{pc['f1']:.4f}", pc['support']])
            writer.writerow(row)

        # Compute mean/std across folds (excluding ensemble)
        fold_only = [fold_metrics[k] for k in fold_metrics if k.startswith('fold_') and fold_metrics[k] is not None]

        if len(fold_only) > 1:
            metrics_keys = ['accuracy', 'balanced_accuracy', 'kappa', 'mcc', 'macro_f1', 'weighted_f1']

            # Mean row
            mean_row = ['Mean', '', '', '']
            for k in metrics_keys:
                values = [m[k] for m in fold_only]
                mean_row.append(f"{np.mean(values):.4f}")
            for cls in CLASS_NAMES:
                for metric in ['precision', 'recall', 'f1']:
                    values = [m['per_class'][cls][metric] for m in fold_only]
                    mean_row.append(f"{np.mean(values):.4f}")
                mean_row.append('')  # Support column
            writer.writerow(mean_row)

            # Std row
            std_row = ['Std', '', '', '']
            for k in metrics_keys:
                values = [m[k] for m in fold_only]
                std_row.append(f"{np.std(values):.4f}")
            for cls in CLASS_NAMES:
                for metric in ['precision', 'recall', 'f1']:
                    values = [m['per_class'][cls][metric] for m in fold_only]
                    std_row.append(f"{np.std(values):.4f}")
                std_row.append('')  # Support column
            writer.writerow(std_row)

    print(f"Patch metrics summary saved: {summary_file}")


def process_dataset_model(predictions_base: Path, dataset_id: int, model: str,
                          min_cancer_threshold: float = 0.10, suffix: str = ""):
    """Process a single dataset/model combination."""
    dataset_name = DATASETS.get(dataset_id)
    if not dataset_name:
        print(f"Unknown dataset ID: {dataset_id}")
        return

    dataset_full_name = f"Dataset{dataset_id}_{dataset_name}"
    model_dir = predictions_base / dataset_full_name / model

    if not model_dir.exists():
        print(f"Predictions not found: {model_dir}")
        return

    # Labels directory
    labels_dir = Path("nnUNet_raw") / dataset_full_name / "labelsTs"
    if not labels_dir.exists():
        print(f"Labels not found: {labels_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Processing: {dataset_full_name} / {model}")
    print(f"{'='*80}")

    fold_metrics = {}

    # Process each fold
    for fold_dir in sorted(model_dir.iterdir()):
        if not fold_dir.is_dir():
            continue

        fold_name = fold_dir.name
        if not (fold_name.startswith('fold_') or fold_name == 'ensemble'):
            continue

        print(f"  Processing {fold_name}...")

        metrics = compute_patch_metrics_for_fold(fold_dir, labels_dir, min_cancer_threshold)

        if metrics:
            fold_metrics[fold_name] = metrics
            save_patch_metrics_report(fold_dir, metrics, fold_name, suffix)
            print(f"    Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        else:
            print(f"    No predictions found")

    # Create summary
    if fold_metrics:
        create_patch_metrics_summary(model_dir, fold_metrics, suffix)

    return fold_metrics


def create_dataset_cm_summary(predictions_base: Path, all_results: Dict, suffix: str = ""):
    """Create per-dataset confusion matrix summary across all models."""
    # Group by dataset
    by_dataset = {}
    for (dataset_id, model), fold_metrics in all_results.items():
        if dataset_id not in by_dataset:
            by_dataset[dataset_id] = {}

        # Prefer ensemble, fallback to summing fold CMs
        cm = None
        metrics = None
        if 'ensemble' in fold_metrics and fold_metrics['ensemble'] is not None:
            metrics = fold_metrics['ensemble']
            cm = np.array(metrics['confusion_matrix'])
        else:
            fold_cms = [
                np.array(v['confusion_matrix'])
                for k, v in fold_metrics.items()
                if k.startswith('fold_') and v is not None and 'confusion_matrix' in v
            ]
            if fold_cms:
                cm = sum(fold_cms)

        if cm is not None:
            by_dataset[dataset_id][model] = {'confusion_matrix': cm, 'metrics': metrics}

    # Write per-dataset summary files
    for dataset_id in sorted(by_dataset.keys()):
        dataset_name = DATASETS.get(dataset_id, 'Unknown')
        dataset_full = f"Dataset{dataset_id}_{dataset_name}"
        models_data = by_dataset[dataset_id]

        output_dir = predictions_base / dataset_full
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_file = output_dir / f"confusion_matrices_summary{suffix}.txt"
        with open(txt_file, 'w') as f:
            f.write(f"CONFUSION MATRICES SUMMARY - {dataset_full}\n")
            f.write("=" * 80 + "\n")
            f.write("Rows: True | Columns: Predicted\n")
            f.write("Source: ensemble (or summed across folds)\n\n")

            for model in sorted(models_data.keys()):
                data = models_data[model]
                cm = data['confusion_matrix']
                metrics = data['metrics']

                f.write(f"Model: {model}\n")
                f.write("-" * 63 + "\n")
                f.write(f"{'':15s}")
                for name in CLASS_NAMES:
                    f.write(f"{name:>12s}")
                f.write("\n")

                for i, name in enumerate(CLASS_NAMES):
                    f.write(f"{name:15s}")
                    for j in range(4):
                        f.write(f"{int(cm[i, j]):12d}")
                    f.write("\n")

                if metrics:
                    f.write(f"\nAccuracy: {metrics['accuracy']:.4f}  "
                            f"Balanced Acc: {metrics['balanced_accuracy']:.4f}  "
                            f"Macro F1: {metrics['macro_f1']:.4f}\n")

                f.write("\n")

            f.write("=" * 80 + "\n")

        print(f"Confusion matrix summary saved: {txt_file}")


def main():
    parser = argparse.ArgumentParser(description='Compute per-patch classification metrics')

    parser.add_argument('--dataset_id', type=int,
                       help='Dataset ID (100-104)')
    parser.add_argument('--model', type=str, choices=MODELS,
                       help='Model name')
    parser.add_argument('--all_models', action='store_true',
                       help='Process all models for specified dataset')
    parser.add_argument('--all', action='store_true',
                       help='Process all datasets and models')
    parser.add_argument('--predictions_dir', type=str, default='predictions_classification',
                       help='Base predictions directory')

    args = parser.parse_args()

    predictions_base = Path(args.predictions_dir)

    if not predictions_base.exists():
        print(f"Predictions directory not found: {predictions_base}")
        return 1

    thresholds = [(0.0, '_thr_0'), (0.10, '_thr_10')]

    for threshold, suffix in thresholds:
        print(f"\n{'#'*80}")
        print(f"Running with threshold: {threshold*100:.0f}% (suffix: {suffix})")
        print(f"{'#'*80}")

        all_results = {}

        if args.all:
            # Process all datasets and models
            for dataset_id in DATASETS.keys():
                for model in MODELS:
                    result = process_dataset_model(predictions_base, dataset_id, model,
                                                   threshold, suffix)
                    if result:
                        all_results[(dataset_id, model)] = result

        elif args.dataset_id and args.all_models:
            # Process all models for one dataset
            for model in MODELS:
                result = process_dataset_model(predictions_base, args.dataset_id, model,
                                               threshold, suffix)
                if result:
                    all_results[(args.dataset_id, model)] = result

        elif args.dataset_id and args.model:
            # Process specific dataset/model
            result = process_dataset_model(predictions_base, args.dataset_id, args.model,
                                           threshold, suffix)
            if result:
                all_results[(args.dataset_id, args.model)] = result

        else:
            # Auto-detect and process all available
            print("No specific dataset/model specified. Processing all available...")
            for dataset_dir in sorted(predictions_base.iterdir()):
                if not dataset_dir.is_dir() or not dataset_dir.name.startswith('Dataset'):
                    continue

                # Parse dataset ID
                try:
                    dataset_id = int(dataset_dir.name.split('_')[0].replace('Dataset', ''))
                except ValueError:
                    continue

                for model_dir in sorted(dataset_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name
                    if model in MODELS:
                        result = process_dataset_model(predictions_base, dataset_id, model,
                                                       threshold, suffix)
                        if result:
                            all_results[(dataset_id, model)] = result

        # Create per-dataset confusion matrix summary
        if all_results:
            create_dataset_cm_summary(predictions_base, all_results, suffix)

    print("\n" + "="*80)
    print("PATCH METRICS COMPUTATION COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
