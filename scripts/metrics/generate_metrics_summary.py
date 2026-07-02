#!/usr/bin/env python3
"""
Generate Summary Metrics Across Multiple Folds
Aggregates metrics from individual fold evaluation reports and computes mean ± stddev.

Usage:
    python generate_metrics_summary.py --results_dir predictions/Dataset104_test --output summary.csv
    python generate_metrics_summary.py --results_dir predictions/Dataset104_test --include_ensemble
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsSummaryGenerator:
    """Generate summary statistics across multiple folds"""

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.fold_dirs = []
        self.ensemble_dir = None

    def find_fold_directories(self, include_ensemble=False):
        """Find all fold directories and optionally ensemble directory"""
        # Find fold_X directories
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name.startswith('fold_'):
                json_file = item / 'test_evaluation_report.json'
                if json_file.exists():
                    self.fold_dirs.append(item)

        self.fold_dirs.sort(key=lambda x: int(x.name.split('_')[1]))

        # Find ensemble directory
        if include_ensemble:
            ensemble_dir = self.results_dir / 'ensemble'
            if ensemble_dir.exists():
                json_file = ensemble_dir / 'test_evaluation_report.json'
                if json_file.exists():
                    self.ensemble_dir = ensemble_dir

        logger.info(f"Found {len(self.fold_dirs)} fold directories")
        if include_ensemble and self.ensemble_dir:
            logger.info(f"Found ensemble directory")

        return len(self.fold_dirs) > 0

    def load_fold_metrics(self, fold_dir):
        """Load metrics from a fold directory"""
        json_file = fold_dir / 'test_evaluation_report.json'

        with open(json_file, 'r') as f:
            data = json.load(f)

        return data

    def aggregate_metrics(self, include_ensemble=False):
        """Aggregate metrics across all folds"""
        all_metrics = []
        fold_names = []

        # Load metrics from each fold
        for fold_dir in self.fold_dirs:
            fold_name = fold_dir.name
            metrics = self.load_fold_metrics(fold_dir)
            all_metrics.append(metrics)
            fold_names.append(fold_name)
            logger.info(f"Loaded metrics from {fold_name}")

        # Load ensemble metrics if available
        ensemble_metrics = None
        if include_ensemble and self.ensemble_dir:
            ensemble_metrics = self.load_fold_metrics(self.ensemble_dir)
            logger.info("Loaded metrics from ensemble")

        return all_metrics, fold_names, ensemble_metrics

    def create_summary_dataframe(self, all_metrics, fold_names):
        """Create a summary DataFrame with per-fold metrics and statistics"""
        rows = []

        # Extract key metrics from each fold
        for fold_name, metrics in zip(fold_names, all_metrics):
            agg = metrics['aggregate']
            row = {
                'Fold': fold_name,
                'Accuracy': agg['accuracy'],
                'Balanced_Accuracy': agg['balanced_accuracy'],
                'Cohen_Kappa': agg['cohen_kappa'],
                'MCC': agg['mcc'],
                'Foreground_Macro_Dice': agg.get('foreground_macro_dice', np.nan),
                'Foreground_Macro_Precision': agg.get('foreground_macro_precision', np.nan),
                'Foreground_Macro_Recall': agg.get('foreground_macro_recall', np.nan),
                'Foreground_Macro_F1': agg.get('foreground_macro_f1', np.nan),
            }

            # Add per-class Dice scores
            for class_name, class_metrics in agg['per_class'].items():
                row[f'{class_name}_Dice'] = class_metrics['dice']
                row[f'{class_name}_Precision'] = class_metrics['precision']
                row[f'{class_name}_Recall'] = class_metrics['recall']
                row[f'{class_name}_F1'] = class_metrics['f1']

            rows.append(row)

        df = pd.DataFrame(rows)

        # Compute mean and stddev
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        mean_row = df[numeric_cols].mean().to_dict()
        mean_row['Fold'] = 'Mean'

        std_row = df[numeric_cols].std().to_dict()
        std_row['Fold'] = 'Std Dev'

        # Add mean and stddev rows
        df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        return df

    def add_ensemble_to_summary(self, df, ensemble_metrics):
        """Add ensemble metrics as a separate row"""
        if ensemble_metrics is None:
            return df

        agg = ensemble_metrics['aggregate']
        row = {
            'Fold': 'Ensemble',
            'Accuracy': agg['accuracy'],
            'Balanced_Accuracy': agg['balanced_accuracy'],
            'Cohen_Kappa': agg['cohen_kappa'],
            'MCC': agg['mcc'],
            'Foreground_Macro_Dice': agg.get('foreground_macro_dice', np.nan),
            'Foreground_Macro_Precision': agg.get('foreground_macro_precision', np.nan),
            'Foreground_Macro_Recall': agg.get('foreground_macro_recall', np.nan),
            'Foreground_Macro_F1': agg.get('foreground_macro_f1', np.nan),
        }

        # Add per-class metrics
        for class_name, class_metrics in agg['per_class'].items():
            row[f'{class_name}_Dice'] = class_metrics['dice']
            row[f'{class_name}_Precision'] = class_metrics['precision']
            row[f'{class_name}_Recall'] = class_metrics['recall']
            row[f'{class_name}_F1'] = class_metrics['f1']

        # Add ensemble row after mean/std
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        return df

    def generate_summary(self, output_file, include_ensemble=False):
        """Generate complete summary and save to CSV"""
        if not self.find_fold_directories(include_ensemble):
            logger.error("No fold directories found!")
            return None

        # Aggregate metrics
        all_metrics, fold_names, ensemble_metrics = self.aggregate_metrics(include_ensemble)

        # Create summary DataFrame
        summary_df = self.create_summary_dataframe(all_metrics, fold_names)

        # Add ensemble if available
        if include_ensemble and ensemble_metrics:
            summary_df = self.add_ensemble_to_summary(summary_df, ensemble_metrics)

        # Save to CSV
        summary_df.to_csv(output_file, index=False)
        logger.info(f"Summary saved to {output_file}")

        # Also create a transposed version for easier reading
        transposed_file = Path(output_file).parent / f"{Path(output_file).stem}_transposed.csv"
        summary_df_t = summary_df.set_index('Fold').T
        summary_df_t.to_csv(transposed_file)
        logger.info(f"Transposed summary saved to {transposed_file}")

        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS ACROSS FOLDS")
        print("="*80)

        mean_row = summary_df[summary_df['Fold'] == 'Mean'].iloc[0]
        std_row = summary_df[summary_df['Fold'] == 'Std Dev'].iloc[0]

        print("\nKey Metrics (Mean ± Std Dev):")
        print("-"*80)
        for col in ['Accuracy', 'Foreground_Macro_Dice', 'Metastatic_Dice', 'HCC_Dice', 'CHO_Dice']:
            if col in mean_row:
                mean_val = mean_row[col]
                std_val = std_row[col]
                print(f"{col:30s}: {mean_val:.4f} ± {std_val:.4f}")

        if include_ensemble and ensemble_metrics:
            print("\nEnsemble Metrics:")
            print("-"*80)
            ensemble_row = summary_df[summary_df['Fold'] == 'Ensemble'].iloc[0]
            for col in ['Accuracy', 'Foreground_Macro_Dice', 'Metastatic_Dice', 'HCC_Dice', 'CHO_Dice']:
                if col in ensemble_row:
                    print(f"{col:30s}: {ensemble_row[col]:.4f}")

        print("="*80)

        return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Generate summary metrics across multiple folds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summary from fold results
  python generate_metrics_summary.py --results_dir predictions/Dataset104_test --output summary.csv

  # Include ensemble metrics
  python generate_metrics_summary.py --results_dir predictions/Dataset104_test --output summary.csv --include_ensemble
        """
    )

    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing fold_X subdirectories with test_evaluation_report.json files')
    parser.add_argument('--output', type=str, default='metrics_summary.csv',
                       help='Output CSV file path (default: metrics_summary.csv)')
    parser.add_argument('--include_ensemble', action='store_true',
                       help='Include ensemble metrics if available')

    args = parser.parse_args()

    # Create summary generator
    generator = MetricsSummaryGenerator(args.results_dir)

    # Generate summary
    summary_df = generator.generate_summary(args.output, args.include_ensemble)

    if summary_df is not None:
        logger.info("\n✅ Summary generation completed successfully!")
    else:
        logger.error("\n❌ Summary generation failed!")


if __name__ == "__main__":
    main()
