#!/usr/bin/env python3
"""
Compute diagnosis-level classification metrics from segmentation predictions
Determines diagnosis based on majority tumor class (excluding background)

Usage:
    python compute_diagnosis_metrics.py \
        --predictions /path/to/predictions \
        --labels /path/to/labels \
        --output diagnosis_report.txt
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

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
from PIL import Image


class DiagnosisMetricsCalculator:
    """Calculate diagnosis-level metrics from segmentation predictions"""

    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.class_names = ["Background", "Metastatic", "HCC", "CHO"]
        self.diagnosis_names = ["Healthy", "Metastatic", "HCC", "CHO"]
        self.foreground_classes = [1, 2, 3]  # Tumor classes

    def determine_diagnosis(self, mask: np.ndarray) -> int:
        """
        Determine diagnosis based on majority tumor class (excluding background)

        Args:
            mask: Segmentation mask with class indices (0=background, 1-3=tumor types)

        Returns:
            diagnosis:
                0 = Healthy (no tumor pixels)
                1 = Metastatic (class 1 has most tumor pixels)
                2 = HCC (class 2 has most tumor pixels)
                3 = CHO (class 3 has most tumor pixels)
        """
        # Count pixels for each class
        counts = np.bincount(mask.flatten(), minlength=self.num_classes)

        # Get tumor class counts (exclude background)
        tumor_counts = counts[1:]  # [metastatic, HCC, CHO]

        # If no tumor pixels, diagnosis is Healthy
        if tumor_counts.sum() == 0:
            return 0

        # Otherwise, diagnosis is the tumor class with most pixels
        # +1 because we excluded background (class 0)
        diagnosis = np.argmax(tumor_counts) + 1

        return diagnosis

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """Load a segmentation mask from file (PNG or NPY)"""
        if mask_path.suffix == '.png':
            mask = np.array(Image.open(mask_path))
        elif mask_path.suffix == '.npy':
            mask = np.load(mask_path)
            # Handle one-hot encoded labels
            if mask.ndim == 3:
                if mask.shape[2] == self.num_classes:
                    mask = np.argmax(mask, axis=2)
                elif mask.shape[0] == self.num_classes:
                    mask = np.argmax(mask, axis=0)
        else:
            raise ValueError(f"Unsupported file format: {mask_path.suffix}")

        return mask.astype(np.uint8)

    def compute_metrics(self, predictions_dir: Path, labels_dir: Path) -> Dict:
        """
        Compute diagnosis metrics for all predictions

        Args:
            predictions_dir: Directory containing prediction masks
            labels_dir: Directory containing ground truth labels

        Returns:
            Dictionary with all metrics and confusion matrix
        """
        print(f"Loading predictions from: {predictions_dir}")
        print(f"Loading labels from: {labels_dir}")

        # Find all prediction files
        pred_files = sorted(list(predictions_dir.glob("*.png")) + list(predictions_dir.glob("*.npy")))

        if len(pred_files) == 0:
            raise ValueError(f"No prediction files found in {predictions_dir}")

        print(f"Found {len(pred_files)} prediction files")

        # Collect diagnoses
        pred_diagnoses = []
        true_diagnoses = []
        case_names = []
        pixel_counts = []

        for pred_file in pred_files:
            case_name = pred_file.stem

            # Find corresponding label file
            label_file = labels_dir / pred_file.name
            if not label_file.exists():
                # Try different extension
                if pred_file.suffix == '.png':
                    label_file = labels_dir / f"{case_name}.npy"
                else:
                    label_file = labels_dir / f"{case_name}.png"

            if not label_file.exists():
                print(f"Warning: Label not found for {case_name}, skipping")
                continue

            # Load masks
            pred_mask = self.load_mask(pred_file)
            true_mask = self.load_mask(label_file)

            # Check dimensions match
            if pred_mask.shape != true_mask.shape:
                print(f"Warning: Shape mismatch for {case_name}: "
                      f"pred={pred_mask.shape}, true={true_mask.shape}, skipping")
                continue

            # Determine diagnoses
            pred_diag = self.determine_diagnosis(pred_mask)
            true_diag = self.determine_diagnosis(true_mask)

            pred_diagnoses.append(pred_diag)
            true_diagnoses.append(true_diag)
            case_names.append(case_name)

            # Count pixels per class for statistics
            pred_counts = np.bincount(pred_mask.flatten(), minlength=self.num_classes)
            true_counts = np.bincount(true_mask.flatten(), minlength=self.num_classes)
            pixel_counts.append({
                'case': case_name,
                'true_diagnosis': self.diagnosis_names[true_diag],
                'pred_diagnosis': self.diagnosis_names[pred_diag],
                'pred_counts': pred_counts.tolist(),
                'true_counts': true_counts.tolist()
            })

        if len(pred_diagnoses) == 0:
            raise ValueError("No valid prediction-label pairs found")

        print(f"\nProcessed {len(pred_diagnoses)} cases")

        # Convert to numpy arrays
        pred_diagnoses = np.array(pred_diagnoses)
        true_diagnoses = np.array(true_diagnoses)

        # Calculate metrics
        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = accuracy_score(true_diagnoses, pred_diagnoses)
        metrics['balanced_accuracy'] = balanced_accuracy_score(true_diagnoses, pred_diagnoses)

        # Confusion matrix
        cm = confusion_matrix(true_diagnoses, pred_diagnoses, labels=[0, 1, 2, 3])
        metrics['confusion_matrix'] = cm

        # Per-class metrics
        precision = precision_score(true_diagnoses, pred_diagnoses,
                                   average=None, labels=[0, 1, 2, 3], zero_division=0)
        recall = recall_score(true_diagnoses, pred_diagnoses,
                            average=None, labels=[0, 1, 2, 3], zero_division=0)
        f1 = f1_score(true_diagnoses, pred_diagnoses,
                     average=None, labels=[0, 1, 2, 3], zero_division=0)

        metrics['per_diagnosis'] = {}
        for i, diag_name in enumerate(self.diagnosis_names):
            support = int(np.sum(true_diagnoses == i))
            metrics['per_diagnosis'][diag_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': support
            }

        # Macro-averaged metrics
        metrics['macro_precision'] = float(np.mean(precision))
        metrics['macro_recall'] = float(np.mean(recall))
        metrics['macro_f1'] = float(np.mean(f1))

        # Weighted metrics (by support)
        metrics['weighted_precision'] = precision_score(true_diagnoses, pred_diagnoses,
                                                       average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(true_diagnoses, pred_diagnoses,
                                                  average='weighted', zero_division=0)
        metrics['weighted_f1'] = f1_score(true_diagnoses, pred_diagnoses,
                                         average='weighted', zero_division=0)

        # Cohen's Kappa
        metrics['kappa'] = cohen_kappa_score(true_diagnoses, pred_diagnoses)

        # Matthews correlation coefficient
        metrics['mcc'] = matthews_corrcoef(true_diagnoses, pred_diagnoses)

        # Store case-level information
        metrics['num_cases'] = len(pred_diagnoses)
        metrics['case_details'] = pixel_counts

        # Classification report
        metrics['classification_report'] = classification_report(
            true_diagnoses, pred_diagnoses,
            target_names=self.diagnosis_names,
            labels=[0, 1, 2, 3],
            zero_division=0
        )

        return metrics

    def print_report(self, metrics: Dict):
        """Print formatted metrics report"""
        print("\n" + "="*80)
        print("DIAGNOSIS CLASSIFICATION REPORT")
        print("="*80)
        print(f"Total cases: {metrics['num_cases']}")
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
        print(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}")

        print("\n" + "-"*80)
        print("CONFUSION MATRIX")
        print("-"*80)
        print("Rows: True diagnosis | Columns: Predicted diagnosis")
        print(f"Classes: {' | '.join(self.diagnosis_names)}")
        print()

        cm = metrics['confusion_matrix']
        # Print header
        print(f"{'':15s}", end="")
        for name in self.diagnosis_names:
            print(f"{name:>12s}", end="")
        print()
        print("-" * 63)

        # Print rows
        for i, name in enumerate(self.diagnosis_names):
            print(f"{name:15s}", end="")
            for j in range(len(self.diagnosis_names)):
                print(f"{cm[i, j]:12d}", end="")
            print()

        print("\n" + "-"*80)
        print("PER-DIAGNOSIS METRICS")
        print("-"*80)
        print(f"{'Diagnosis':<15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        print("-" * 63)

        for diag_name in self.diagnosis_names:
            metrics_diag = metrics['per_diagnosis'][diag_name]
            print(f"{diag_name:<15s} "
                  f"{metrics_diag['precision']:>10.4f} "
                  f"{metrics_diag['recall']:>10.4f} "
                  f"{metrics_diag['f1']:>10.4f} "
                  f"{metrics_diag['support']:>10d}")

        print("-" * 63)
        print(f"{'Macro Avg':<15s} "
              f"{metrics['macro_precision']:>10.4f} "
              f"{metrics['macro_recall']:>10.4f} "
              f"{metrics['macro_f1']:>10.4f} "
              f"{metrics['num_cases']:>10d}")
        print(f"{'Weighted Avg':<15s} "
              f"{metrics['weighted_precision']:>10.4f} "
              f"{metrics['weighted_recall']:>10.4f} "
              f"{metrics['weighted_f1']:>10.4f} "
              f"{metrics['num_cases']:>10d}")

        print("\n" + "-"*80)
        print("SKLEARN CLASSIFICATION REPORT")
        print("-"*80)
        print(metrics['classification_report'])

        print("="*80)

    def save_report(self, metrics: Dict, output_file: Path, predictions_dir: Path, labels_dir: Path):
        """Save detailed report to file"""
        with open(output_file, 'w') as f:
            f.write("DIAGNOSIS CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Predictions directory: {predictions_dir}\n")
            f.write(f"Labels directory: {labels_dir}\n")
            f.write(f"Total cases analyzed: {metrics['num_cases']}\n")
            f.write(f"Number of classes: {self.num_classes}\n")
            f.write(f"Classes: {', '.join(self.class_names)}\n")
            f.write(f"Diagnoses: {', '.join(self.diagnosis_names)}\n")
            f.write("="*80 + "\n\n")

            f.write("DIAGNOSIS CRITERIA:\n")
            f.write("-"*80 + "\n")
            f.write("• Healthy: No tumor pixels detected (all background)\n")
            f.write("• Metastatic: Majority of tumor pixels are class 1\n")
            f.write("• HCC: Majority of tumor pixels are class 2\n")
            f.write("• CHO: Majority of tumor pixels are class 3\n")
            f.write("\n")

            f.write("OVERALL METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:                {metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy:       {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Cohen's Kappa:           {metrics['kappa']:.4f}\n")
            f.write(f"Matthews Correlation:    {metrics['mcc']:.4f}\n")
            f.write(f"Macro Precision:         {metrics['macro_precision']:.4f}\n")
            f.write(f"Macro Recall:            {metrics['macro_recall']:.4f}\n")
            f.write(f"Macro F1:                {metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted Precision:      {metrics['weighted_precision']:.4f}\n")
            f.write(f"Weighted Recall:         {metrics['weighted_recall']:.4f}\n")
            f.write(f"Weighted F1:             {metrics['weighted_f1']:.4f}\n")
            f.write("\n")

            f.write("CONFUSION MATRIX:\n")
            f.write("-"*80 + "\n")
            f.write("Rows: True diagnosis | Columns: Predicted diagnosis\n")
            f.write(f"Classes: {' | '.join(self.diagnosis_names)}\n\n")

            cm = metrics['confusion_matrix']
            # Header
            f.write(f"{'':15s}")
            for name in self.diagnosis_names:
                f.write(f"{name:>12s}")
            f.write("\n" + "-" * 63 + "\n")

            # Rows
            for i, name in enumerate(self.diagnosis_names):
                f.write(f"{name:15s}")
                for j in range(len(self.diagnosis_names)):
                    f.write(f"{cm[i, j]:12d}")
                f.write("\n")
            f.write("\n")

            f.write("PER-DIAGNOSIS METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Diagnosis':<15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}\n")
            f.write("-" * 63 + "\n")

            for diag_name in self.diagnosis_names:
                metrics_diag = metrics['per_diagnosis'][diag_name]
                f.write(f"{diag_name:<15s} "
                       f"{metrics_diag['precision']:>10.4f} "
                       f"{metrics_diag['recall']:>10.4f} "
                       f"{metrics_diag['f1']:>10.4f} "
                       f"{metrics_diag['support']:>10d}\n")

            f.write("-" * 63 + "\n")
            f.write(f"{'Macro Avg':<15s} "
                   f"{metrics['macro_precision']:>10.4f} "
                   f"{metrics['macro_recall']:>10.4f} "
                   f"{metrics['macro_f1']:>10.4f} "
                   f"{metrics['num_cases']:>10d}\n")
            f.write(f"{'Weighted Avg':<15s} "
                   f"{metrics['weighted_precision']:>10.4f} "
                   f"{metrics['weighted_recall']:>10.4f} "
                   f"{metrics['weighted_f1']:>10.4f} "
                   f"{metrics['num_cases']:>10d}\n")
            f.write("\n")

            f.write("SKLEARN CLASSIFICATION REPORT:\n")
            f.write("-"*80 + "\n")
            f.write(metrics['classification_report'])
            f.write("\n")

            f.write("="*80 + "\n")
            f.write("INTERPRETATION:\n")
            f.write("-"*80 + "\n")
            f.write("• Diagnosis accuracy shows how well predictions identify\n")
            f.write("  the primary tumor type in each image\n")
            f.write("• Balanced accuracy accounts for class imbalance\n")
            f.write("• Cohen's Kappa measures agreement beyond chance\n")
            f.write("• Macro avg: unweighted mean across all diagnoses\n")
            f.write("• Weighted avg: weighted by support (number of cases)\n")
            f.write("="*80 + "\n")

        print(f"\nDetailed report saved to: {output_file}")

        # Also save JSON
        json_file = output_file.with_suffix('.json')

        # Convert numpy arrays to lists for JSON serialization
        metrics_json = metrics.copy()
        metrics_json['confusion_matrix'] = metrics['confusion_matrix'].tolist()

        with open(json_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        print(f"JSON metrics saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute diagnosis-level classification metrics from segmentation predictions'
    )

    parser.add_argument('--predictions', type=str, required=True,
                       help='Directory containing prediction masks (PNG or NPY)')
    parser.add_argument('--labels', type=str, required=True,
                       help='Directory containing ground truth labels (PNG or NPY)')
    parser.add_argument('--output', type=str, default='diagnosis_report.txt',
                       help='Output file for report (default: diagnosis_report.txt)')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of classes including background (default: 4)')

    args = parser.parse_args()

    # Convert paths
    predictions_dir = Path(args.predictions)
    labels_dir = Path(args.labels)
    output_file = Path(args.output)

    # Validate directories
    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        return 1

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return 1

    # Create calculator
    calculator = DiagnosisMetricsCalculator(num_classes=args.num_classes)

    # Compute metrics
    try:
        metrics = calculator.compute_metrics(predictions_dir, labels_dir)

        # Print report to console
        calculator.print_report(metrics)

        # Save detailed report to file
        calculator.save_report(metrics, output_file, predictions_dir, labels_dir)

        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
