#!/usr/bin/env python3
"""
Aggregate classification metrics summaries into Excel files per Dataset.
Creates two .xlsx files per Dataset:
  - diagnosis_metrics_all_models.xlsx  (from metrics_summary.csv per model)
  - patch_metrics_all_models.xlsx      (from patch_metrics_summary.csv per model)
Each model gets its own sheet in the workbook.

Usage:
    # Process all datasets
    python aggregate_classification_metrics.py

    # Process specific dataset
    python aggregate_classification_metrics.py --dataset_id 104

    # Custom predictions directory
    python aggregate_classification_metrics.py --predictions_dir predictions_classification
"""

import argparse
import sys
import io
from pathlib import Path

import pandas as pd

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATASETS = {
    100: 'Liver1',
    101: 'Liver2',
    102: 'Liver3',
    103: 'Liver4',
    104: 'Liver5'
}

MODELS = ['convnextv2', 'efficientnetv2', 'swinv2', 'maxvit', 'densenet']


def aggregate_dataset(dataset_dir: Path):
    """Aggregate metrics for a single dataset directory into xlsx files."""
    dataset_name = dataset_dir.name
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*80}")

    diagnosis_sheets = {}
    patch_suffixes = ['_thr_0', '_thr_10']
    patch_sheets_by_suffix = {s: {} for s in patch_suffixes}

    for model_dir in sorted(dataset_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name not in MODELS:
            continue

        model = model_dir.name

        # Read diagnosis metrics summary
        diagnosis_csv = model_dir / "metrics_summary.csv"
        if diagnosis_csv.exists():
            df = pd.read_csv(diagnosis_csv)
            diagnosis_sheets[model] = df
            print(f"  {model}: diagnosis metrics loaded ({len(df)} rows)")
        else:
            print(f"  {model}: metrics_summary.csv not found")

        # Read patch metrics summaries for each threshold
        for suffix in patch_suffixes:
            patch_csv = model_dir / f"patch_metrics_summary{suffix}.csv"
            if patch_csv.exists():
                df = pd.read_csv(patch_csv)
                patch_sheets_by_suffix[suffix][model] = df
                print(f"  {model}: patch metrics {suffix} loaded ({len(df)} rows)")
            else:
                print(f"  {model}: patch_metrics_summary{suffix}.csv not found")

    # Write diagnosis metrics xlsx
    if diagnosis_sheets:
        output_file = dataset_dir / "diagnosis_metrics_all_models.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for model, df in diagnosis_sheets.items():
                df.to_excel(writer, sheet_name=model, index=False)
        print(f"\n  Saved: {output_file} ({len(diagnosis_sheets)} sheets)")
    else:
        print(f"\n  No diagnosis metrics found for {dataset_name}")

    # Write patch metrics xlsx for each threshold
    for suffix in patch_suffixes:
        patch_sheets = patch_sheets_by_suffix[suffix]
        if patch_sheets:
            output_file = dataset_dir / f"patch_metrics_all_models{suffix}.xlsx"
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for model, df in patch_sheets.items():
                    df.to_excel(writer, sheet_name=model, index=False)
            print(f"  Saved: {output_file} ({len(patch_sheets)} sheets)")
        else:
            print(f"  No patch metrics {suffix} found for {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate classification metrics into Excel files per Dataset'
    )
    parser.add_argument('--dataset_id', type=int,
                       help='Process specific dataset (100-104)')
    parser.add_argument('--predictions_dir', type=str, default='predictions_classification',
                       help='Base predictions directory')

    args = parser.parse_args()
    predictions_base = Path(args.predictions_dir)

    if not predictions_base.exists():
        print(f"Predictions directory not found: {predictions_base}")
        return 1

    if args.dataset_id:
        dataset_name = DATASETS.get(args.dataset_id)
        if not dataset_name:
            print(f"Unknown dataset ID: {args.dataset_id}")
            return 1
        dataset_dir = predictions_base / f"Dataset{args.dataset_id}_{dataset_name}"
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return 1
        aggregate_dataset(dataset_dir)
    else:
        # Process all datasets
        for dataset_dir in sorted(predictions_base.iterdir()):
            if dataset_dir.is_dir() and dataset_dir.name.startswith('Dataset'):
                aggregate_dataset(dataset_dir)

    print(f"\n{'='*80}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*80}")
    return 0


if __name__ == "__main__":
    exit(main())
