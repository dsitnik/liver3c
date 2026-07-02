#!/usr/bin/env python3
"""
Compute metrics for nnU-Net test set predictions
Evaluates predictions against ground truth labels with no information leakage.

Usage: python compute_test_metrics.py --predictions test_predictions/2d --labels nnUNet_raw/Dataset101_LiverCancerTest/labelsTs
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import json
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    jaccard_score,
    confusion_matrix
)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSetEvaluator:
    def __init__(self, predictions_dir, labels_dir, num_classes=4):
        self.predictions_dir = Path(predictions_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self.class_names = ['Background', 'Metastatic', 'HCC', 'CHO']
        self.foreground_classes = [1, 2, 3]

    def load_case(self, case_id):
        """Load prediction and ground truth for a case"""
        # Load prediction
        pred_file = self.predictions_dir / f"{case_id}.png"
        if not pred_file.exists():
            logger.warning(f"Prediction not found for {case_id}")
            return None, None

        pred = np.array(Image.open(pred_file))

        # Load ground truth
        label_file = self.labels_dir / f"{case_id}.png"
        if not label_file.exists():
            logger.warning(f"Label not found for {case_id}")
            return None, None

        label = np.array(Image.open(label_file))

        return pred, label

    def calculate_dice_coefficient(self, y_true, y_pred, class_idx):
        """Calculate Dice coefficient for a specific class"""
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)

        intersection = np.sum(y_true_binary * y_pred_binary)
        union = np.sum(y_true_binary) + np.sum(y_pred_binary)

        if union == 0:
            return 1.0 if np.sum(y_true_binary) == 0 else 0.0

        dice = 2.0 * intersection / union
        return dice

    def evaluate_all_cases(self):
        """Evaluate all test cases"""
        # Find all prediction files
        pred_files = sorted(self.predictions_dir.glob("*.png"))

        if len(pred_files) == 0:
            logger.error(f"No prediction files found in {self.predictions_dir}")
            return None

        all_results = []
        all_preds = []
        all_labels = []

        logger.info(f"Found {len(pred_files)} test cases to evaluate")

        for pred_file in pred_files:
            case_id = pred_file.stem
            pred, label = self.load_case(case_id)

            if pred is None or label is None:
                continue

            # Flatten for sklearn metrics
            pred_flat = pred.flatten()
            label_flat = label.flatten()

            all_preds.extend(pred_flat)
            all_labels.extend(label_flat)

            # Calculate per-case metrics
            case_metrics = self.calculate_case_metrics(pred_flat, label_flat, case_id)
            all_results.append(case_metrics)

        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )

        return {
            'per_case': all_results,
            'aggregate': aggregate_metrics,
            'num_cases': len(all_results)
        }

    def calculate_case_metrics(self, y_true, y_pred, case_id):
        """Calculate metrics for a single case"""
        metrics = {'case_id': case_id}

        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Per-class Dice
        for class_idx in range(self.num_classes):
            dice = self.calculate_dice_coefficient(y_true, y_pred, class_idx)
            metrics[f'{self.class_names[class_idx]}_Dice'] = dice

        # Foreground-only metrics
        y_true_fg = y_true[y_true > 0]
        y_pred_fg = y_pred[y_true > 0]

        if len(y_true_fg) > 0:
            metrics['foreground_accuracy'] = accuracy_score(y_true_fg, y_pred_fg)
        else:
            metrics['foreground_accuracy'] = np.nan

        return metrics

    def calculate_aggregate_metrics(self, y_true, y_pred):
        """Calculate aggregate metrics across all cases"""
        metrics = {}

        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        metrics['confusion_matrix'] = cm.tolist()

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0
        )

        per_class_metrics = {}
        for i in range(self.num_classes):
            class_metrics = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i]),
                'dice': self.calculate_dice_coefficient(y_true, y_pred, i),
                'jaccard': jaccard_score(y_true == i, y_pred == i, zero_division=0)
            }
            per_class_metrics[self.class_names[i]] = class_metrics

        metrics['per_class'] = per_class_metrics

        # Foreground-only metrics (excluding background)
        foreground_mask = y_true > 0
        y_true_fg = y_true[foreground_mask]
        y_pred_fg = y_pred[foreground_mask]

        if len(y_true_fg) > 0:
            # Macro-averaged metrics for foreground classes
            fg_precision, fg_recall, fg_f1, _ = precision_recall_fscore_support(
                y_true_fg, y_pred_fg,
                labels=self.foreground_classes,
                average='macro',
                zero_division=0
            )

            metrics['foreground_macro_precision'] = fg_precision
            metrics['foreground_macro_recall'] = fg_recall
            metrics['foreground_macro_f1'] = fg_f1

            # Micro-averaged metrics for foreground classes
            fg_precision_micro, fg_recall_micro, fg_f1_micro, _ = precision_recall_fscore_support(
                y_true_fg, y_pred_fg,
                labels=self.foreground_classes,
                average='micro',
                zero_division=0
            )

            metrics['foreground_micro_precision'] = fg_precision_micro
            metrics['foreground_micro_recall'] = fg_recall_micro
            metrics['foreground_micro_f1'] = fg_f1_micro

            # Foreground Dice scores
            dice_scores = [per_class_metrics[self.class_names[i]]['dice']
                          for i in self.foreground_classes]
            metrics['foreground_macro_dice'] = np.mean(dice_scores)

        return metrics

    def generate_report(self, results, output_file=None):
        """Generate a comprehensive evaluation report"""
        if results is None:
            logger.error("No results to report")
            return

        report = []
        report.append("="*80)
        report.append("TEST SET EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Number of test cases: {results['num_cases']}")
        report.append("")

        # Aggregate metrics
        agg = results['aggregate']
        report.append("OVERALL METRICS:")
        report.append("-"*40)
        report.append(f"Accuracy: {agg['accuracy']:.4f}")
        report.append(f"Balanced Accuracy: {agg['balanced_accuracy']:.4f}")
        report.append(f"Cohen's Kappa: {agg['cohen_kappa']:.4f}")
        report.append(f"Matthews Correlation Coefficient: {agg['mcc']:.4f}")
        report.append("")

        # Foreground metrics
        report.append("FOREGROUND-ONLY METRICS (Tumor Classes):")
        report.append("-"*40)
        report.append(f"Macro Dice: {agg.get('foreground_macro_dice', 0):.4f}")
        report.append(f"Macro Precision: {agg.get('foreground_macro_precision', 0):.4f}")
        report.append(f"Macro Recall: {agg.get('foreground_macro_recall', 0):.4f}")
        report.append(f"Macro F1: {agg.get('foreground_macro_f1', 0):.4f}")
        report.append("")
        report.append(f"Micro Precision: {agg.get('foreground_micro_precision', 0):.4f}")
        report.append(f"Micro Recall: {agg.get('foreground_micro_recall', 0):.4f}")
        report.append(f"Micro F1: {agg.get('foreground_micro_f1', 0):.4f}")
        report.append("")

        # Per-class metrics
        report.append("PER-CLASS METRICS:")
        report.append("-"*40)
        for class_name, class_metrics in agg['per_class'].items():
            report.append(f"\n{class_name}:")
            report.append(f"  Dice Score: {class_metrics['dice']:.4f}")
            report.append(f"  Jaccard Index: {class_metrics['jaccard']:.4f}")
            report.append(f"  Precision: {class_metrics['precision']:.4f}")
            report.append(f"  Recall: {class_metrics['recall']:.4f}")
            report.append(f"  F1 Score: {class_metrics['f1']:.4f}")
            report.append(f"  Support: {class_metrics['support']:,} pixels")

        # Confusion matrix
        report.append("\nCONFUSION MATRIX:")
        report.append("-"*40)
        cm = np.array(agg['confusion_matrix'])

        # Create formatted confusion matrix
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {name}" for name in self.class_names],
            columns=[f"Pred {name}" for name in self.class_names]
        )
        report.append(str(cm_df))
        report.append("")

        # Per-case summary statistics
        case_df = pd.DataFrame(results['per_case'])
        dice_cols = [f'{name}_Dice' for name in self.class_names]

        report.append("\nPER-CASE STATISTICS:")
        report.append("-"*40)
        for col in ['accuracy'] + dice_cols:
            if col in case_df.columns:
                mean_val = case_df[col].mean()
                std_val = case_df[col].std()
                min_val = case_df[col].min()
                max_val = case_df[col].max()
                report.append(f"{col}: {mean_val:.4f} ± {std_val:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")

        # Join report lines
        report_text = "\n".join(report)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")

        # Also save detailed results as JSON
        json_file = Path(output_file).with_suffix('.json') if output_file else 'test_results.json'
        with open(json_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {
                'aggregate': {k: (v.tolist() if isinstance(v, np.ndarray) else
                                 float(v) if isinstance(v, np.floating) else v)
                             for k, v in results['aggregate'].items()},
                'num_cases': results['num_cases'],
                'per_case_summary': {
                    'accuracy_mean': float(case_df['accuracy'].mean()),
                    'accuracy_std': float(case_df['accuracy'].std()),
                    'foreground_dice_mean': float(case_df[[f'{name}_Dice' for name in self.class_names[1:]]].mean().mean()),
                    'foreground_dice_std': float(case_df[[f'{name}_Dice' for name in self.class_names[1:]]].mean().std())
                }
            }
            json.dump(json_results, f, indent=2)
        logger.info(f"Detailed results saved to {json_file}")

        # Export per-case metrics to CSV
        per_case_csv = Path(output_file).parent / f"{Path(output_file).stem}_per_case.csv" if output_file else 'test_results_per_case.csv'
        case_df.to_csv(per_case_csv, index=False)
        logger.info(f"Per-case metrics saved to {per_case_csv}")

        # Export aggregate metrics to CSV
        aggregate_csv = Path(output_file).parent / f"{Path(output_file).stem}_aggregate.csv" if output_file else 'test_results_aggregate.csv'
        self.export_aggregate_to_csv(results['aggregate'], aggregate_csv)
        logger.info(f"Aggregate metrics saved to {aggregate_csv}")

        print(report_text)
        return report_text

    def export_aggregate_to_csv(self, aggregate_metrics, output_file):
        """Export aggregate metrics to CSV format"""
        rows = []

        # Overall metrics
        rows.append(['Overall Metrics', ''])
        rows.append(['Accuracy', aggregate_metrics['accuracy']])
        rows.append(['Balanced Accuracy', aggregate_metrics['balanced_accuracy']])
        rows.append(['Cohen Kappa', aggregate_metrics['cohen_kappa']])
        rows.append(['MCC', aggregate_metrics['mcc']])
        rows.append(['', ''])

        # Foreground metrics
        rows.append(['Foreground Metrics', ''])
        rows.append(['Foreground Macro Dice', aggregate_metrics.get('foreground_macro_dice', '')])
        rows.append(['Foreground Macro Precision', aggregate_metrics.get('foreground_macro_precision', '')])
        rows.append(['Foreground Macro Recall', aggregate_metrics.get('foreground_macro_recall', '')])
        rows.append(['Foreground Macro F1', aggregate_metrics.get('foreground_macro_f1', '')])
        rows.append(['Foreground Micro Precision', aggregate_metrics.get('foreground_micro_precision', '')])
        rows.append(['Foreground Micro Recall', aggregate_metrics.get('foreground_micro_recall', '')])
        rows.append(['Foreground Micro F1', aggregate_metrics.get('foreground_micro_f1', '')])
        rows.append(['', ''])

        # Per-class metrics header
        rows.append(['Per-Class Metrics', 'Dice', 'Jaccard', 'Precision', 'Recall', 'F1', 'Support'])

        for class_name, metrics in aggregate_metrics['per_class'].items():
            rows.append([
                class_name,
                metrics['dice'],
                metrics['jaccard'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['support']
            ])

        # Write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False, header=False)

        return output_file

def main():
    parser = argparse.ArgumentParser(description='Evaluate nnU-Net test set predictions')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Directory containing prediction PNG files')
    parser.add_argument('--labels', type=str, required=True,
                       help='Directory containing ground truth label PNG files')
    parser.add_argument('--output', type=str, default='test_evaluation_report.txt',
                       help='Output file for evaluation report')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of classes including background (default: 4)')
    parser.add_argument('--checkpoint_type', type=str, choices=['best', 'final'],
                       help='Type of checkpoint being evaluated (for report naming)')

    args = parser.parse_args()

    logger.info(f"Evaluating predictions from: {args.predictions}")
    logger.info(f"Using labels from: {args.labels}")
    if args.checkpoint_type:
        logger.info(f"Checkpoint type: {args.checkpoint_type}")

    # Create evaluator
    evaluator = TestSetEvaluator(
        predictions_dir=args.predictions,
        labels_dir=args.labels,
        num_classes=args.num_classes
    )

    # Evaluate all cases
    results = evaluator.evaluate_all_cases()

    # Generate report
    if results:
        # Add checkpoint type to results if specified
        if args.checkpoint_type:
            results['checkpoint_type'] = args.checkpoint_type

        evaluator.generate_report(results, output_file=args.output)
        logger.info("Evaluation completed successfully!")

        # Print key metrics for easy comparison
        logger.info("\n" + "="*50)
        logger.info("KEY METRICS SUMMARY:")
        logger.info(f"Accuracy: {results['aggregate']['accuracy']:.4f}")
        logger.info(f"Foreground Macro Dice: {results['aggregate'].get('foreground_macro_dice', 0):.4f}")
        logger.info(f"Foreground Macro F1: {results['aggregate'].get('foreground_macro_f1', 0):.4f}")
        logger.info("="*50)
    else:
        logger.error("Evaluation failed - no results generated")

if __name__ == "__main__":
    main()