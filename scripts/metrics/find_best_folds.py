"""
Script to find the best fold for each dataset in nnUNet_results based on validation Dice score.
"""

import json
import math
from pathlib import Path


def find_best_folds(nnunet_results_dir: str) -> dict:
    """
    Find the best fold for each dataset based on validation Dice score.

    Args:
        nnunet_results_dir: Path to nnUNet_results directory

    Returns:
        Dictionary with results for each dataset
    """
    results_path = Path(nnunet_results_dir)
    results = {}

    # Iterate through all datasets
    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith("Dataset"):
            continue

        dataset_name = dataset_dir.name
        results[dataset_name] = {}

        # Iterate through all configurations (trainers)
        for config_dir in sorted(dataset_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            config_name = config_dir.name
            fold_results = []

            # Iterate through all folds
            for fold_dir in sorted(config_dir.iterdir()):
                if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                    continue

                fold_num = int(fold_dir.name.split("_")[1])
                summary_path = fold_dir / "validation" / "summary.json"

                if not summary_path.exists():
                    print(f"Warning: No summary.json found for {dataset_name}/{config_name}/fold_{fold_num}")
                    continue

                try:
                    with open(summary_path, "r") as f:
                        summary = json.load(f)

                    foreground_dice = summary.get("foreground_mean", {}).get("Dice")
                    foreground_iou = summary.get("foreground_mean", {}).get("IoU")

                    # Skip if dice is None or NaN
                    if foreground_dice is None or (isinstance(foreground_dice, float) and math.isnan(foreground_dice)):
                        print(f"Warning: Invalid Dice for {dataset_name}/{config_name}/fold_{fold_num}")
                        continue

                    # Handle NaN IoU
                    if foreground_iou is not None and isinstance(foreground_iou, float) and math.isnan(foreground_iou):
                        foreground_iou = None

                    # Extract per-class dice, filtering out NaN values
                    per_class_dice = {}
                    for k, v in summary.get("mean", {}).items():
                        class_dice = v.get("Dice")
                        if class_dice is not None and not (isinstance(class_dice, float) and math.isnan(class_dice)):
                            per_class_dice[k] = class_dice

                    fold_results.append({
                        "fold": fold_num,
                        "dice": foreground_dice,
                        "iou": foreground_iou,
                        "per_class_dice": per_class_dice
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {summary_path}: {e}")
                    continue

            if fold_results:
                # Sort by dice score (descending) and get the best
                fold_results.sort(key=lambda x: x["dice"], reverse=True)
                best_fold = fold_results[0]

                results[dataset_name][config_name] = {
                    "best_fold": best_fold["fold"],
                    "best_dice": best_fold["dice"],
                    "best_iou": best_fold["iou"],
                    "best_per_class_dice": best_fold["per_class_dice"],
                    "all_folds": [
                        {"fold": f["fold"], "dice": f["dice"], "iou": f["iou"]}
                        for f in fold_results
                    ]
                }

    return results


def main():
    # Resolve the nnUNet_results directory relative to the project root.
    # Run this script from the project root (e.g. python scripts/metrics/find_best_folds.py).
    nnunet_results_dir = Path.cwd() / "nnUNet_results"

    if not nnunet_results_dir.exists():
        print(f"Error: nnUNet_results directory not found at {nnunet_results_dir}")
        return

    print(f"Searching for best folds in: {nnunet_results_dir}")

    # Find best folds
    results = find_best_folds(str(nnunet_results_dir))

    # Print summary
    print("\n" + "=" * 60)
    print("BEST FOLDS SUMMARY")
    print("=" * 60)

    for dataset_name, configs in results.items():
        print(f"\n{dataset_name}:")
        for config_name, data in configs.items():
            print(f"  {config_name}:")
            print(f"    Best Fold: {data['best_fold']}")
            print(f"    Best Dice: {data['best_dice']:.4f}")
            iou_str = f"{data['best_iou']:.4f}" if data['best_iou'] is not None else "N/A"
            print(f"    Best IoU:  {iou_str}")
            print(f"    Per-class Dice: {data['best_per_class_dice']}")

    # Save results
    output_path = nnunet_results_dir / "best_folds.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
