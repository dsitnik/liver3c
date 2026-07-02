#!/usr/bin/env python3
"""
nnUNet Inference Script for Test Set
Runs inference on test data using trained nnU-Net models with various options.

Usage: python nnunet_inference.py --dataset_id 100 --input_dir data/test_images --output_dir predictions/test
"""

import os
import json
import argparse
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class nnUNetInference:
    """nnUNet inference manager for test set evaluation"""

    def __init__(self, dataset_id: int, dataset_name: str = "LiverCancer"):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.dataset_full_name = f"Dataset{dataset_id:03d}_{dataset_name}"

        # Set up environment
        self.setup_environment()

    def setup_environment(self):
        """Setup nnUNet environment variables"""
        current_dir = Path.cwd()
        self.nnunet_raw = current_dir / "nnUNet_raw"
        self.nnunet_preprocessed = current_dir / "nnUNet_preprocessed"
        self.nnunet_results = current_dir / "nnUNet_results"

        # Set environment variables
        os.environ['nnUNet_raw'] = str(self.nnunet_raw)
        os.environ['nnUNet_preprocessed'] = str(self.nnunet_preprocessed)
        os.environ['nnUNet_results'] = str(self.nnunet_results)

        self.dataset_path = self.nnunet_raw / self.dataset_full_name
        self.preprocessed_path = self.nnunet_preprocessed / self.dataset_full_name
        self.results_path = self.nnunet_results / self.dataset_full_name

        logger.info("nnUNet environment setup:")
        logger.info(f"  nnUNet_raw: {self.nnunet_raw}")
        logger.info(f"  nnUNet_preprocessed: {self.nnunet_preprocessed}")
        logger.info(f"  nnUNet_results: {self.nnunet_results}")

    def run_command(self, cmd: List[str], description: str, capture_output: bool = False):
        """Execute a shell command with logging"""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return result.stdout, result.stderr
            else:
                subprocess.run(cmd, check=True)
                return None, None
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if capture_output and e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            if capture_output and e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            raise

    def get_available_configurations(self) -> List[str]:
        """Get available trained configurations"""
        configs = []

        if self.results_path.exists():
            for item in self.results_path.iterdir():
                if item.is_dir() and "nnUNetTrainer" in item.name:
                    # Extract configuration from folder name
                    parts = item.name.split("__")
                    if len(parts) >= 3:
                        config = parts[2]
                        configs.append(config)

        logger.info(f"Available trained configurations: {configs}")
        return configs

    def get_trained_folds(self, configuration: str, trainer: str = None) -> List[int]:
        """Get list of trained folds for a configuration"""
        trained_folds = []
        trainer_name = trainer if trainer else "nnUNetTrainer"
        config_path = self.results_path / f"{trainer_name}__nnUNetPlans__{configuration}"

        if config_path.exists():
            for fold_dir in config_path.iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
                    fold_num = int(fold_dir.name.split("_")[1])
                    # Check if checkpoint exists
                    checkpoint_best = fold_dir / "checkpoint_best.pth"
                    checkpoint_final = fold_dir / "checkpoint_final.pth"
                    if checkpoint_best.exists() or checkpoint_final.exists():
                        trained_folds.append(fold_num)

        trained_folds.sort()
        logger.info(f"Trained folds for {configuration} (trainer: {trainer_name}): {trained_folds}")
        return trained_folds

    def find_best_checkpoint(self, configuration: str) -> Optional[str]:
        """Find the best performing configuration/fold based on validation metrics"""
        config_path = self.results_path / f"nnUNetTrainer__nnUNetPlans__{configuration}"
        best_fold = None
        best_dice = -1

        # Check validation scores for each fold
        for fold_dir in config_path.glob("fold_*"):
            validation_file = fold_dir / "validation" / "summary.json"
            if validation_file.exists():
                try:
                    with open(validation_file, 'r') as f:
                        summary = json.load(f)
                        # Look for mean Dice score
                        if "mean" in summary and "Dice" in summary["mean"]:
                            dice_score = summary["mean"]["Dice"]
                            if dice_score > best_dice:
                                best_dice = dice_score
                                best_fold = int(fold_dir.name.split("_")[1])
                except:
                    pass

        if best_fold is not None:
            logger.info(f"Best fold for {configuration}: fold {best_fold} (Dice: {best_dice:.4f})")

        return best_fold

    def run_inference_single_fold(self, input_dir: Path, output_dir: Path,
                                 configuration: str, fold: int,
                                 save_probabilities: bool = False,
                                 disable_tta: bool = False,
                                 step_size: float = 0.5,
                                 use_gpu: bool = True,
                                 checkpoint: str = "checkpoint_best",
                                 trainer: str = None):
        """Run inference using a single fold"""
        logger.info(f"Running inference with {configuration} fold {fold}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", str(self.dataset_id),
            "-c", configuration,
            "-f", str(fold),
            "-chk", checkpoint
        ]

        if trainer:
            cmd.extend(["-tr", trainer])

        if save_probabilities:
            cmd.append("--save_probabilities")

        if disable_tta:
            cmd.append("--disable_tta")

        cmd.extend(["-step_size", str(step_size)])

        if not use_gpu:
            cmd.extend(["-device", "cpu"])

        self.run_command(cmd, f"Inference for {configuration} fold {fold}")
        logger.info(f"Predictions saved to: {output_dir}")

    def run_inference_ensemble(self, input_dir: Path, output_dir: Path,
                             configuration: str, folds: Optional[List[int]] = None,
                             save_probabilities: bool = False,
                             disable_tta: bool = False,
                             step_size: float = 0.5,
                             use_gpu: bool = True,
                             checkpoint: str = "checkpoint_best",
                             trainer: str = None):
        """Run ensemble inference using multiple folds"""
        if folds is None:
            folds = self.get_trained_folds(configuration, trainer)

        if not folds:
            raise ValueError(f"No trained folds found for {configuration}")

        logger.info(f"Running ensemble inference with {configuration} using folds: {folds}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", str(self.dataset_id),
            "-c", configuration,
            "-f"
        ] + [str(f) for f in folds] + [
            "-chk", checkpoint
        ]

        if trainer:
            cmd.extend(["-tr", trainer])

        if save_probabilities:
            cmd.append("--save_probabilities")

        if disable_tta:
            cmd.append("--disable_tta")

        cmd.extend(["-step_size", str(step_size)])

        if not use_gpu:
            cmd.extend(["-device", "cpu"])

        self.run_command(cmd, f"Ensemble inference for {configuration}")
        logger.info(f"Ensemble predictions saved to: {output_dir}")

    def run_best_ensemble(self, input_dir: Path, output_dir: Path,
                         save_probabilities: bool = False,
                         disable_tta: bool = False,
                         step_size: float = 0.5,
                         use_gpu: bool = True):
        """Run inference using the best ensemble configuration found by nnUNet"""
        logger.info("Running inference with best ensemble configuration")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Use nnUNet's apply_best_ensemble
        cmd = [
            "nnUNetv2_apply_best_ensemble",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", str(self.dataset_id)
        ]

        if save_probabilities:
            cmd.append("--save_probabilities")

        if disable_tta:
            cmd.append("--disable_tta")

        cmd.extend(["-step_size", str(step_size)])

        if not use_gpu:
            cmd.extend(["-device", "cpu"])

        try:
            self.run_command(cmd, "Best ensemble inference")
            logger.info(f"Best ensemble predictions saved to: {output_dir}")
        except subprocess.CalledProcessError as e:
            logger.warning("Best ensemble not available, falling back to available configurations")
            # Fallback to manual ensemble
            configs = self.get_available_configurations()
            if configs:
                self.run_inference_ensemble(input_dir, output_dir, configs[0],
                                          save_probabilities=save_probabilities,
                                          disable_tta=disable_tta,
                                          step_size=step_size,
                                          use_gpu=use_gpu)

    def convert_predictions_to_numpy(self, predictions_dir: Path, output_dir: Path):
        """Convert prediction PNG/nifti files back to numpy arrays"""
        logger.info(f"Converting predictions from {predictions_dir} to numpy format")

        output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image
        import numpy as np

        prediction_files = sorted(predictions_dir.glob("*.png"))
        if not prediction_files:
            prediction_files = sorted(predictions_dir.glob("*.nii.gz"))

        for pred_file in prediction_files:
            if pred_file.suffix == ".png":
                # Load PNG
                img = Image.open(pred_file)
                pred_array = np.array(img)
            else:
                # Load NIfTI
                import nibabel as nib
                img = nib.load(pred_file)
                pred_array = img.get_fdata().astype(np.uint8)

            # Save as numpy
            output_file = output_dir / f"{pred_file.stem}.npy"
            np.save(output_file, pred_array)
            logger.info(f"Converted: {pred_file.name} -> {output_file.name}")

        logger.info(f"Converted {len(prediction_files)} predictions to numpy format")

    def visualize_predictions(self, predictions_dir: Path, output_dir: Path):
        """Create RGB visualizations of predictions with color-coded classes"""
        logger.info(f"Creating visualizations from {predictions_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image
        import numpy as np

        # Define colors for each class (RGB)
        color_map = {
            0: [0, 0, 0],       # Background - Black
            1: [255, 0, 0],     # Metastatic - Red
            2: [0, 255, 0],     # HCC - Green
            3: [0, 0, 255]      # CHO - Blue
        }

        prediction_files = sorted(predictions_dir.glob("*.png"))
        if not prediction_files:
            prediction_files = sorted(predictions_dir.glob("*.nii.gz"))

        for pred_file in prediction_files:
            if pred_file.suffix == ".png":
                # Load PNG
                img = Image.open(pred_file)
                pred_array = np.array(img)
            else:
                # Load NIfTI
                import nibabel as nib
                img = nib.load(pred_file)
                pred_array = img.get_fdata().astype(np.uint8)

            # Create RGB image
            h, w = pred_array.shape[:2]
            rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

            # Apply colors to each class
            for class_idx, color in color_map.items():
                mask = pred_array == class_idx
                rgb_image[mask] = color

            # Save visualization
            output_file = output_dir / f"{pred_file.stem}_viz.png"
            Image.fromarray(rgb_image).save(output_file)
            logger.info(f"Visualized: {pred_file.name} -> {output_file.name}")

        logger.info(f"Created {len(prediction_files)} visualizations")

    def export_to_mat(self, predictions_dir: Path, output_dir: Path):
        """Export predictions to MATLAB .mat format"""
        logger.info(f"Exporting predictions to .mat format from {predictions_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from scipy.io import savemat
        except ImportError:
            logger.error("scipy is required for .mat export. Install with: pip install scipy")
            return

        from PIL import Image
        import numpy as np

        prediction_files = sorted(predictions_dir.glob("*.png"))
        if not prediction_files:
            prediction_files = sorted(predictions_dir.glob("*.nii.gz"))

        for pred_file in prediction_files:
            if pred_file.suffix == ".png":
                # Load PNG
                img = Image.open(pred_file)
                pred_array = np.array(img)
            else:
                # Load NIfTI
                import nibabel as nib
                img = nib.load(pred_file)
                pred_array = img.get_fdata().astype(np.uint8)

            # Save as .mat
            output_file = output_dir / f"{pred_file.stem}.mat"
            savemat(output_file, {'prediction': pred_array})
            logger.info(f"Exported: {pred_file.name} -> {output_file.name}")

        logger.info(f"Exported {len(prediction_files)} predictions to .mat format")

    def create_inference_summary(self, input_dir: Path, output_dir: Path,
                                configuration: str, folds: List[int],
                                inference_time: float = None):
        """Create a summary of the inference run"""
        summary = {
            "dataset": {
                "id": self.dataset_id,
                "name": self.dataset_name,
                "full_name": self.dataset_full_name
            },
            "inference": {
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "configuration": configuration,
                "folds_used": folds,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        if inference_time:
            summary["inference"]["time_seconds"] = inference_time

        # Count predictions
        pred_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.nii.gz"))
        summary["inference"]["num_predictions"] = len(pred_files)

        summary_file = output_dir / "inference_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Inference summary saved: {summary_file}")
        return summary

def main():
    parser = argparse.ArgumentParser(description='nnUNet Inference Script for Test Set')
    parser.add_argument('--dataset_id', type=int, required=True,
                       help='Dataset ID (3-digit number)')
    parser.add_argument('--dataset_name', type=str, default='LiverCancer',
                       help='Dataset name')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save predictions')
    parser.add_argument('--configuration', type=str,
                       choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'],
                       help='Specific configuration to use (default: auto-detect best)')
    parser.add_argument('--fold', type=int,
                       help='Use specific fold for inference')
    parser.add_argument('--folds', nargs='+', type=int,
                       help='Use specific folds for ensemble (e.g., --folds 0 1 2 3)')
    parser.add_argument('--use_best', action='store_true',
                       help='Use best ensemble configuration (default)')
    parser.add_argument('--all_folds', action='store_true',
                       help='Use all available trained folds for ensemble')
    parser.add_argument('--save_probabilities', action='store_true',
                       help='Save softmax probabilities')
    parser.add_argument('--disable_tta', action='store_true',
                       help='Disable test time augmentation')
    parser.add_argument('--step_size', type=float, default=0.5,
                       help='Step size for sliding window (default: 0.5)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pth',
                       help='Which checkpoint to use (default: checkpoint_best.pth)')
    parser.add_argument('--trainer', type=str,
                       help='Custom trainer class name (e.g., nnUNetTrainer_500epochs)')
    parser.add_argument('--convert_to_numpy', action='store_true',
                       help='Convert predictions to numpy format')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--compute_metrics', action='store_true',
                       help='Automatically compute test metrics after inference')
    parser.add_argument('--labels_dir', type=str,
                       help='Directory containing ground truth labels (for metrics computation)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create RGB visualizations of predictions (Red=Metastatic, Green=HCC, Blue=CHO)')
    parser.add_argument('--export_mat', action='store_true',
                       help='Export predictions to MATLAB .mat format')
    parser.add_argument('--generate_summary', action='store_true',
                       help='Generate summary CSV with metrics across all folds (requires --all_folds or --folds)')

    args = parser.parse_args()

    # Initialize inference manager
    inference = nnUNetInference(args.dataset_id, args.dataset_name)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Determine Python executable (use .venv on Windows)
    if sys.platform == "win32":
        python_exe = str(Path(".venv/Scripts/python.exe"))
    else:
        python_exe = "python"

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    try:
        import time
        start_time = time.time()

        if args.use_best or (not args.configuration and not args.fold and not args.folds):
            # Use best ensemble by default
            inference.run_best_ensemble(
                input_dir, output_dir,
                save_probabilities=args.save_probabilities,
                disable_tta=args.disable_tta,
                step_size=args.step_size,
                use_gpu=not args.cpu
            )
            configuration = "best_ensemble"
            folds = "auto"

        elif args.fold is not None:
            # Single fold inference
            if not args.configuration:
                configs = inference.get_available_configurations()
                if not configs:
                    logger.error("No trained configurations found")
                    sys.exit(1)
                configuration = configs[0]
            else:
                configuration = args.configuration

            inference.run_inference_single_fold(
                input_dir, output_dir,
                configuration, args.fold,
                save_probabilities=args.save_probabilities,
                disable_tta=args.disable_tta,
                step_size=args.step_size,
                use_gpu=not args.cpu,
                checkpoint=args.checkpoint,
                trainer=args.trainer
            )
            folds = [args.fold]

        else:
            # Ensemble inference
            if not args.configuration:
                configs = inference.get_available_configurations()
                if not configs:
                    logger.error("No trained configurations found")
                    sys.exit(1)
                configuration = configs[0]
                logger.info(f"Auto-selected configuration: {configuration}")
            else:
                configuration = args.configuration

            if args.folds:
                folds = args.folds
            elif args.all_folds:
                folds = inference.get_trained_folds(configuration, args.trainer)
            else:
                # Try to find best fold or use all
                best_fold = inference.find_best_checkpoint(configuration)
                if best_fold is not None:
                    folds = [best_fold]
                    logger.info(f"Using best performing fold: {best_fold}")
                else:
                    folds = inference.get_trained_folds(configuration, args.trainer)

            # If all_folds is set, first test each fold individually
            if args.all_folds:
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing each fold individually (found {len(folds)} folds)")
                logger.info(f"{'='*60}\n")

                for fold_idx in folds:
                    fold_output_dir = output_dir / f"fold_{fold_idx}"
                    logger.info(f"\n--- Testing fold {fold_idx} ---")

                    inference.run_inference_single_fold(
                        input_dir, fold_output_dir,
                        configuration, fold_idx,
                        save_probabilities=args.save_probabilities,
                        disable_tta=args.disable_tta,
                        step_size=args.step_size,
                        use_gpu=not args.cpu,
                        checkpoint=args.checkpoint,
                        trainer=args.trainer
                    )

                    # Post-process individual fold
                    if args.visualize:
                        viz_output = fold_output_dir / "visualizations"
                        inference.visualize_predictions(fold_output_dir, viz_output)

                    if args.export_mat:
                        mat_output = fold_output_dir / "mat"
                        inference.export_to_mat(fold_output_dir, mat_output)

                    if args.compute_metrics:
                        if args.labels_dir:
                            labels_dir = Path(args.labels_dir)
                        else:
                            labels_dir = inference.nnunet_raw / inference.dataset_full_name / "labelsTs"

                        if labels_dir.exists():
                            metrics_output = fold_output_dir / "test_evaluation_report.txt"
                            # Extract checkpoint type: checkpoint_best.pth -> best
                            checkpoint_type = args.checkpoint.replace('checkpoint_', '').replace('.pth', '')

                            metrics_cmd = [
                                python_exe, "compute_test_metrics.py",
                                "--predictions", str(fold_output_dir),
                                "--labels", str(labels_dir),
                                "--output", str(metrics_output),
                                "--checkpoint_type", checkpoint_type
                            ]

                            inference.run_command(metrics_cmd, f"Computing metrics for fold {fold_idx}")
                            logger.info(f"Fold {fold_idx} metrics saved to: {metrics_output}")

                logger.info(f"\n{'='*60}")
                logger.info("Now creating ensemble prediction from all folds")
                logger.info(f"{'='*60}\n")

                # Create ensemble output directory (preserve original for summary generation)
                base_output_dir = output_dir  # Save original directory
                ensemble_output_dir = output_dir / "ensemble"
                output_dir = ensemble_output_dir

            inference.run_inference_ensemble(
                input_dir, output_dir,
                configuration, folds,
                save_probabilities=args.save_probabilities,
                disable_tta=args.disable_tta,
                step_size=args.step_size,
                use_gpu=not args.cpu,
                checkpoint=args.checkpoint,
                trainer=args.trainer
            )

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")

        # Create summary
        if isinstance(folds, str):
            folds = []  # For best ensemble
        inference.create_inference_summary(input_dir, output_dir,
                                          configuration, folds, inference_time)

        # Convert to numpy if requested
        if args.convert_to_numpy:
            numpy_output = output_dir.parent / f"{output_dir.name}_numpy"
            inference.convert_predictions_to_numpy(output_dir, numpy_output)

        # Create visualizations if requested
        if args.visualize:
            viz_output = output_dir / "visualizations"
            inference.visualize_predictions(output_dir, viz_output)

        # Export to .mat if requested
        if args.export_mat:
            mat_output = output_dir / "mat"
            inference.export_to_mat(output_dir, mat_output)

        # Compute metrics if requested
        if args.compute_metrics:
            # Determine labels directory
            if args.labels_dir:
                labels_dir = Path(args.labels_dir)
            else:
                # Auto-detect labels directory
                labels_dir = inference.nnunet_raw / inference.dataset_full_name / "labelsTs"

            if labels_dir.exists():
                logger.info(f"\n{'='*60}")
                logger.info("Computing test set metrics...")
                logger.info(f"{'='*60}")

                # Run compute_test_metrics.py
                metrics_output = output_dir / "test_evaluation_report.txt"
                # Extract checkpoint type: checkpoint_best.pth -> best
                checkpoint_type = args.checkpoint.replace('checkpoint_', '').replace('.pth', '')

                metrics_cmd = [
                    python_exe, "compute_test_metrics.py",
                    "--predictions", str(output_dir),
                    "--labels", str(labels_dir),
                    "--output", str(metrics_output),
                    "--checkpoint_type", checkpoint_type
                ]

                inference.run_command(metrics_cmd, "Computing test metrics")
                logger.info(f"Metrics report saved to: {metrics_output}")
            else:
                logger.warning(f"Labels directory not found: {labels_dir}")
                logger.warning("Skipping metrics computation")

        # Generate summary across folds if requested
        if args.generate_summary and (args.all_folds or args.folds):
            logger.info(f"\n{'='*60}")
            logger.info("Generating summary metrics across folds...")
            logger.info(f"{'='*60}")

            # Use base_output_dir if it exists (when all_folds was used), otherwise use output_dir
            results_dir = base_output_dir if 'base_output_dir' in locals() else output_dir
            summary_output = results_dir / "metrics_summary.csv"

            # Call generate_metrics_summary.py
            summary_cmd = [
                python_exe, "generate_metrics_summary.py",
                "--results_dir", str(results_dir),
                "--output", str(summary_output)
            ]

            # Include ensemble if it was created
            if args.all_folds:
                summary_cmd.append("--include_ensemble")

            try:
                inference.run_command(summary_cmd, "Generating metrics summary")
                logger.info(f"Summary saved to: {summary_output}")
            except Exception as e:
                logger.warning(f"Failed to generate summary: {e}")

        logger.info(f"\n{'='*60}")
        logger.info("Inference completed successfully!")
        logger.info(f"Predictions saved to: {output_dir}")
        if args.save_probabilities:
            logger.info("Softmax probabilities also saved")
        if args.visualize:
            logger.info(f"Visualizations saved to: {output_dir}/visualizations/")
        if args.export_mat:
            logger.info(f".mat files saved to: {output_dir}/mat/")
        if args.compute_metrics:
            logger.info("Test metrics computed and saved")
        if args.generate_summary:
            logger.info("Summary metrics across folds generated")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()