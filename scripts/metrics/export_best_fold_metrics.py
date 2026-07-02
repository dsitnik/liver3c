"""
Script to export best fold prediction metrics to a single Excel file for all partitions.
Maps Partition_N to Dataset{99+N}_Liver{N} based on best_folds.json.
"""

import json
from pathlib import Path

import pandas as pd


def load_best_folds(best_folds_path: Path) -> dict:
    """Load the best folds JSON file."""
    with open(best_folds_path, "r") as f:
        return json.load(f)


def get_partition_to_dataset_mapping() -> dict:
    """
    Create mapping from Partition to Dataset.
    Partition_1 -> Dataset100_Liver1
    Partition_2 -> Dataset101_Liver2
    etc.
    """
    return {
        "Partition_1": "Dataset100_Liver1",
        "Partition_2": "Dataset101_Liver2",
        "Partition_3": "Dataset102_Liver3",
        "Partition_4": "Dataset103_Liver4",
        "Partition_5": "Dataset104_Liver5",
    }


def parse_aggregate_csv(csv_path: Path) -> dict:
    """Parse the aggregate CSV file and return metrics as a dictionary."""
    metrics = {}

    with open(csv_path, "r") as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]

        # Detect sections
        if parts[0] == "Overall Metrics":
            current_section = "overall"
            continue
        elif parts[0] == "Foreground Metrics":
            current_section = "foreground"
            continue
        elif parts[0] == "Per-Class Metrics":
            current_section = "per_class"
            continue

        # Parse metrics based on section
        if current_section == "overall" and len(parts) >= 2 and parts[1]:
            metrics[parts[0]] = float(parts[1])
        elif current_section == "foreground" and len(parts) >= 2 and parts[1]:
            metrics[parts[0]] = float(parts[1])
        elif current_section == "per_class" and len(parts) >= 6 and parts[0] not in ["Per-Class Metrics"]:
            class_name = parts[0]
            if parts[1]:  # Has values
                metrics[f"{class_name}_Dice"] = float(parts[1])
                metrics[f"{class_name}_Jaccard"] = float(parts[2])
                metrics[f"{class_name}_Precision"] = float(parts[3])
                metrics[f"{class_name}_Recall"] = float(parts[4])
                metrics[f"{class_name}_F1"] = float(parts[5])
                if len(parts) > 6 and parts[6]:
                    metrics[f"{class_name}_Support"] = int(parts[6])

    return metrics


def get_partition_data(
    partition_name: str,
    dataset_name: str,
    best_fold: int,
    predictions_dir: Path,
    best_dice: float,
    best_iou: float,
) -> dict:
    """Get metrics data for a partition."""

    fold_dir = predictions_dir / partition_name / "complete" / f"fold_{best_fold}"

    if not fold_dir.exists():
        print(f"Warning: Fold directory not found: {fold_dir}")
        return None

    per_case_csv = fold_dir / "test_evaluation_report_per_case.csv"
    aggregate_csv = fold_dir / "test_evaluation_report_aggregate.csv"

    if not per_case_csv.exists():
        print(f"Warning: Per-case CSV not found: {per_case_csv}")
        return None

    # Read per-case metrics
    df_per_case = pd.read_csv(per_case_csv)
    df_per_case.insert(0, "Partition", partition_name)

    # Parse aggregate metrics
    aggregate_metrics = {}
    if aggregate_csv.exists():
        aggregate_metrics = parse_aggregate_csv(aggregate_csv)

    return {
        "partition": partition_name,
        "dataset": dataset_name,
        "best_fold": best_fold,
        "best_dice": best_dice,
        "best_iou": best_iou,
        "aggregate_metrics": aggregate_metrics,
        "per_case_df": df_per_case,
    }


def main():
    # Resolve project dirs relative to the project root.
    # Run this script from the project root (e.g. python scripts/metrics/export_best_fold_metrics.py).
    predictions_dir = Path.cwd() / "predictions"
    nnunet_results_dir = Path.cwd() / "nnUNet_results"
    best_folds_path = nnunet_results_dir / "best_folds.json"

    # Check paths exist
    if not best_folds_path.exists():
        print(f"Error: best_folds.json not found at {best_folds_path}")
        print("Please run find_best_folds.py first.")
        return

    if not predictions_dir.exists():
        print(f"Error: predictions directory not found at {predictions_dir}")
        return

    # Load best folds
    best_folds = load_best_folds(best_folds_path)

    # Get partition to dataset mapping
    partition_mapping = get_partition_to_dataset_mapping()

    # Configuration name (assuming all use the same trainer)
    config_name = "nnUNetTrainer_500epochs__nnUNetPlans__2d"

    print("=" * 60)
    print("EXPORTING BEST FOLD METRICS TO EXCEL")
    print("=" * 60)

    all_partition_data = []

    for partition_name, dataset_name in partition_mapping.items():
        print(f"\nProcessing {partition_name} -> {dataset_name}")

        # Get best fold info for this dataset
        if dataset_name not in best_folds:
            print(f"  Warning: {dataset_name} not found in best_folds.json")
            continue

        if config_name not in best_folds[dataset_name]:
            print(f"  Warning: {config_name} not found for {dataset_name}")
            continue

        fold_info = best_folds[dataset_name][config_name]
        best_fold = fold_info["best_fold"]
        best_dice = fold_info["best_dice"]
        best_iou = fold_info["best_iou"]

        print(f"  Best fold: {best_fold} (Dice: {best_dice:.4f}, IoU: {best_iou:.4f})")

        # Get partition data
        data = get_partition_data(
            partition_name=partition_name,
            dataset_name=dataset_name,
            best_fold=best_fold,
            predictions_dir=predictions_dir,
            best_dice=best_dice,
            best_iou=best_iou,
        )

        if data:
            all_partition_data.append(data)

    if not all_partition_data:
        print("No data found!")
        return

    # Create single Excel file
    output_path = predictions_dir / "all_partitions_best_fold_metrics.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Summary of all partitions
        summary_rows = []
        for data in all_partition_data:
            row = {
                "Partition": data["partition"],
                "Dataset": data["dataset"],
                "Best Fold": data["best_fold"],
                "Validation Dice": data["best_dice"],
                "Validation IoU": data["best_iou"],
            }
            # Add test metrics from aggregate
            agg = data["aggregate_metrics"]
            row["Test Accuracy"] = agg.get("Accuracy")
            row["Test Balanced Accuracy"] = agg.get("Balanced Accuracy")
            row["Test Foreground Macro Dice"] = agg.get("Foreground Macro Dice")
            row["Test Metastatic Dice"] = agg.get("Metastatic_Dice")
            row["Test HCC Dice"] = agg.get("HCC_Dice")
            row["Test CHO Dice"] = agg.get("CHO_Dice")
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Per-class metrics for all partitions
        per_class_rows = []
        classes = ["Background", "Metastatic", "HCC", "CHO"]
        metrics = ["Dice", "Jaccard", "Precision", "Recall", "F1"]

        for data in all_partition_data:
            agg = data["aggregate_metrics"]
            for cls in classes:
                row = {
                    "Partition": data["partition"],
                    "Class": cls,
                }
                for metric in metrics:
                    row[metric] = agg.get(f"{cls}_{metric}")
                row["Support"] = agg.get(f"{cls}_Support")
                per_class_rows.append(row)

        df_per_class = pd.DataFrame(per_class_rows)
        df_per_class.to_excel(writer, sheet_name="Per-Class Metrics", index=False)

        # Sheet 3: All per-case metrics combined
        all_per_case = pd.concat([data["per_case_df"] for data in all_partition_data], ignore_index=True)
        all_per_case.to_excel(writer, sheet_name="Per-Case Metrics", index=False)

        # Sheet 4: Overall metrics comparison
        overall_rows = []
        overall_metrics = ["Accuracy", "Balanced Accuracy", "Cohen Kappa", "MCC",
                          "Foreground Macro Dice", "Foreground Macro Precision",
                          "Foreground Macro Recall", "Foreground Macro F1"]
        for data in all_partition_data:
            agg = data["aggregate_metrics"]
            row = {"Partition": data["partition"]}
            for metric in overall_metrics:
                row[metric] = agg.get(metric)
            overall_rows.append(row)

        df_overall = pd.DataFrame(overall_rows)
        df_overall.to_excel(writer, sheet_name="Overall Metrics", index=False)

    print(f"\n{'=' * 60}")
    print(f"DONE! Created: {output_path}")
    print("=" * 60)
    print("\nSheets in the Excel file:")
    print("  1. Summary - Overview of all partitions with key metrics")
    print("  2. Per-Class Metrics - Dice, Jaccard, Precision, Recall, F1 per class")
    print("  3. Per-Case Metrics - Individual case results for all partitions")
    print("  4. Overall Metrics - Accuracy, Cohen Kappa, MCC, etc.")


if __name__ == "__main__":
    main()
