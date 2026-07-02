#!/usr/bin/env python3
"""
Classification Inference Script for Test Set
Runs inference on test data using trained classification models.
Mirrors nnunet_inference.py structure for consistency.

Features:
- Single fold or multi-fold ensemble inference
- Patch-based classification with majority vote diagnosis
- Grid visualizations with color-coded predictions
- Comprehensive diagnosis-level metrics

Usage:
    python classification_inference.py --dataset_id 104 --model convnextv2 --all_folds --compute_metrics --visualize
"""

import os
import json
import argparse
import sys
import logging
import io
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import Counter

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available, using basic transforms")

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification_inference.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Available timm models for classification (from classification_training.py)
MODELS = {
    'convnextv2': 'convnextv2_base.fcmae_ft_in22k_in1k',
    'efficientnetv2': 'tf_efficientnetv2_m.in21k_ft_in1k',
    'swinv2': 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
    'maxvit': 'maxvit_base_tf_384.in21k_ft_in1k',
    'densenet': 'densenet161.tv_in1k'
}

# Dataset ID to name mapping
DATASETS = {
    100: 'Liver1',
    101: 'Liver2',
    102: 'Liver3',
    103: 'Liver4',
    104: 'Liver5'
}

CLASS_NAMES = {0: 'background', 1: 'metastatic', 2: 'hcc', 3: 'cho'}
DIAGNOSIS_NAMES = ['Healthy', 'Metastatic', 'HCC', 'CHO']

# Color map for visualizations (matching nnunet_inference.py)
COLOR_MAP = {
    0: [0, 0, 0],       # Background - Black
    1: [255, 0, 0],     # Metastatic - Red
    2: [0, 255, 0],     # HCC - Green
    3: [0, 0, 255]      # CHO - Blue
}


def load_normalization_stats(preprocessed_path: Path) -> Tuple[List[float], List[float]]:
    """
    Load dataset-specific normalization statistics from nnU-Net fingerprint.

    Returns:
        Tuple of (mean, std) lists for each channel, normalized to [0, 1] range
    """
    fingerprint_path = preprocessed_path / "dataset_fingerprint.json"

    if not fingerprint_path.exists():
        logger.warning(f"Dataset fingerprint not found: {fingerprint_path}")
        logger.warning("Falling back to simple [0,1] normalization (mean=0.5, std=0.5)")
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    with open(fingerprint_path) as f:
        fingerprint = json.load(f)

    intensity_props = fingerprint.get('foreground_intensity_properties_per_channel', {})

    means = []
    stds = []
    for ch in range(3):
        ch_props = intensity_props.get(str(ch), {})
        # nnU-Net stores raw intensity values (0-255 for 8-bit images)
        # Normalize to [0, 1] range
        mean_val = ch_props.get('mean', 127.5) / 255.0
        std_val = ch_props.get('std', 63.75) / 255.0
        means.append(mean_val)
        stds.append(std_val)

    logger.info(f"Loaded normalization from fingerprint:")
    logger.info(f"  Mean: [{means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f}]")
    logger.info(f"  Std:  [{stds[0]:.4f}, {stds[1]:.4f}, {stds[2]:.4f}]")

    return means, stds


def get_inference_transforms(mean: List[float], std: List[float]):
    """Get inference transforms (only normalization, no augmentation)"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class PatchDataset(Dataset):
    """Dataset for batch inference on extracted patches"""

    def __init__(self, patches: List[np.ndarray], transform=None,
                 normalization_stats: Tuple[List[float], List[float]] = None):
        self.patches = patches
        self.transform = transform
        self.normalization_stats = normalization_stats

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]

        if self.transform is not None:
            transformed = self.transform(image=patch)
            patch_tensor = transformed['image']
        else:
            # Fallback normalization without albumentations
            patch = patch.astype(np.float32) / 255.0
            if self.normalization_stats:
                mean, std = self.normalization_stats
                for c in range(3):
                    patch[:, :, c] = (patch[:, :, c] - mean[c]) / std[c]
            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1))

        return patch_tensor


class ClassificationInference:
    """Classification inference manager for test set evaluation"""

    def __init__(self, dataset_id: int, dataset_name: str, model_name: str):
        """
        Initialize inference manager.

        Args:
            dataset_id: Dataset ID (e.g., 104)
            dataset_name: Dataset name (e.g., 'Liver5')
            model_name: Model name from MODELS dict (e.g., 'convnextv2')
        """
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.dataset_full_name = f"Dataset{dataset_id}_{dataset_name}"
        self.model_name = model_name
        self.model_full_name = MODELS[model_name]

        # Setup paths
        self.setup_paths()

        # Load normalization stats
        self.mean, self.std = load_normalization_stats(self.preprocessed_path)
        self.transform = get_inference_transforms(self.mean, self.std)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def setup_paths(self):
        """Setup directory paths"""
        current_dir = Path.cwd()

        self.nnunet_raw = current_dir / "nnUNet_raw"
        self.nnunet_preprocessed = current_dir / "nnUNet_preprocessed"
        self.classification_results = current_dir / "classification_results"
        self.predictions_dir = current_dir / "predictions_classification"

        self.dataset_path = self.nnunet_raw / self.dataset_full_name
        self.preprocessed_path = self.nnunet_preprocessed / self.dataset_full_name
        self.results_path = self.classification_results / self.dataset_full_name / self.model_name

        logger.info(f"Classification results path: {self.results_path}")

    def set_device(self, use_cpu: bool = False, gpu_id: int = 0):
        """Set computation device"""
        if use_cpu:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")

    def get_trained_folds(self) -> List[int]:
        """Get list of trained folds with checkpoint_best.pth"""
        trained_folds = []

        if self.results_path.exists():
            for fold_dir in self.results_path.iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
                    fold_num = int(fold_dir.name.split("_")[1])
                    checkpoint_path = fold_dir / "checkpoint_best.pth"
                    if checkpoint_path.exists():
                        trained_folds.append(fold_num)

        trained_folds.sort()
        logger.info(f"Trained folds found: {trained_folds}")
        return trained_folds

    def load_model(self, fold: int) -> Tuple[nn.Module, int]:
        """
        Load model from checkpoint_best.pth and return model with patch_size.

        Args:
            fold: Fold number

        Returns:
            Tuple of (model, patch_size)
        """
        fold_path = self.results_path / f"fold_{fold}"
        checkpoint_path = fold_path / "checkpoint_best.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Get patch_size from checkpoint config
        config = checkpoint.get('config', {})
        patch_size = config.get('patch_size', checkpoint.get('patch_size', 384))
        logger.info(f"Patch size from checkpoint: {patch_size}")

        # Create model
        model = timm.create_model(
            self.model_full_name,
            pretrained=False,
            num_classes=4,
            in_chans=3,
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        logger.info(f"Loaded model from fold {fold}, epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Best validation Dice: {checkpoint.get('best_val_dice', 'unknown'):.4f}")

        return model, patch_size

    def load_test_image(self, case_id: str, images_dir: Path) -> np.ndarray:
        """Load 3-channel test image from imagesTs directory"""
        channels = []
        for ch in range(3):
            channel_path = images_dir / f"{case_id}_{ch:04d}.png"
            if not channel_path.exists():
                raise FileNotFoundError(f"Channel file not found: {channel_path}")
            channel = np.array(Image.open(channel_path))
            channels.append(channel)

        # Stack channels: (H, W, 3)
        image = np.stack(channels, axis=-1)
        return image

    def load_test_label(self, case_id: str, labels_dir: Path) -> Optional[np.ndarray]:
        """Load ground truth label from labelsTs directory"""
        label_path = labels_dir / f"{case_id}.png"
        if not label_path.exists():
            return None
        return np.array(Image.open(label_path))

    def extract_patches(self, image: np.ndarray, patch_size: int) -> List[Tuple[np.ndarray, int, int]]:
        """
        Extract non-overlapping patches for inference, including edge patches.
        Edge patches that don't fit the full patch_size are zero-padded.

        Args:
            image: Input image (H, W, 3)
            patch_size: Patch size

        Returns:
            List of (patch, x, y) tuples where x, y are top-left coordinates
        """
        h, w = image.shape[:2]
        patches = []

        # Calculate number of patches needed (ceiling division to include edges)
        n_patches_y = (h + patch_size - 1) // patch_size
        n_patches_x = (w + patch_size - 1) // patch_size

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y = i * patch_size
                x = j * patch_size

                # Extract patch (may be smaller at edges)
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                patch = image[y:y_end, x:x_end, :]

                # Zero-pad if patch is smaller than patch_size
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded_patch

                patches.append((patch, x, y))

        return patches

    def determine_diagnosis(self, patch_predictions: List[int]) -> int:
        """
        Determine diagnosis via majority vote from patch predictions.

        Args:
            patch_predictions: List of predicted class indices for each patch

        Returns:
            Diagnosis: 0=Healthy (all background), 1-3=tumor type with most predictions
        """
        if not patch_predictions:
            return 0  # No patches = Healthy

        # If all patches are background, diagnosis is Healthy
        if all(p == 0 for p in patch_predictions):
            return 0

        # Filter out background predictions for tumor class voting
        tumor_predictions = [p for p in patch_predictions if p > 0]

        if not tumor_predictions:
            return 0  # Healthy

        # Majority vote among tumor classes
        counts = Counter(tumor_predictions)
        diagnosis = counts.most_common(1)[0][0]

        return diagnosis

    def determine_true_diagnosis(self, label: np.ndarray) -> int:
        """
        Determine ground truth diagnosis from label mask.
        Uses same logic as compute_diagnosis_metrics.py
        """
        counts = np.bincount(label.flatten(), minlength=4)
        tumor_counts = counts[1:]  # [metastatic, HCC, CHO]

        if tumor_counts.sum() == 0:
            return 0  # Healthy

        return np.argmax(tumor_counts) + 1

    def run_inference_single_image(self, model: nn.Module, image: np.ndarray,
                                   patch_size: int, batch_size: int = 32) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        """
        Run inference on a single image.

        Returns:
            Tuple of (patch_predictions, patch_info) where patch_info is list of (pred, x, y)
        """
        # Extract patches
        patches_with_coords = self.extract_patches(image, patch_size)

        if not patches_with_coords:
            logger.warning(f"No patches extracted (image too small for patch_size={patch_size})")
            return [], []

        patches = [p[0] for p in patches_with_coords]
        coords = [(p[1], p[2]) for p in patches_with_coords]

        # Create dataset and dataloader
        dataset = PatchDataset(patches, self.transform, (self.mean, self.std))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Run inference
        all_predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                outputs = model(batch)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                all_predictions.extend(predictions.tolist())

        # Combine predictions with coordinates
        patch_info = [(pred, x, y) for pred, (x, y) in zip(all_predictions, coords)]

        return all_predictions, patch_info

    def run_inference_single_fold(self, input_dir: Path, output_dir: Path,
                                  fold: int, batch_size: int = 32,
                                  visualize: bool = False,
                                  compute_metrics: bool = False,
                                  labels_dir: Path = None) -> Dict:
        """
        Run inference using a single fold.

        Returns:
            Dictionary with predictions and optionally metrics
        """
        logger.info(f"Running inference with fold {fold}")

        # Create output directory
        fold_output_dir = output_dir / f"fold_{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        if visualize:
            viz_dir = fold_output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        model, patch_size = self.load_model(fold)

        # Find test images
        test_cases = self._find_test_cases(input_dir)
        logger.info(f"Found {len(test_cases)} test cases")

        # Run inference on each case
        all_predictions = {}
        all_patch_info = {}
        true_diagnoses = {}

        for case_id in test_cases:
            logger.info(f"Processing: {case_id}")

            try:
                # Load image
                image = self.load_test_image(case_id, input_dir)

                # Run inference
                patch_preds, patch_info = self.run_inference_single_image(
                    model, image, patch_size, batch_size
                )

                # Determine diagnosis
                diagnosis = self.determine_diagnosis(patch_preds)
                all_predictions[case_id] = diagnosis
                all_patch_info[case_id] = patch_info

                # Load ground truth if available
                if labels_dir and labels_dir.exists():
                    label = self.load_test_label(case_id, labels_dir)
                    if label is not None:
                        true_diagnoses[case_id] = self.determine_true_diagnosis(label)

                # Save per-case predictions
                self._save_case_predictions(
                    fold_output_dir, case_id, patch_info, diagnosis, patch_size, image.shape
                )

                # Create visualization
                if visualize:
                    true_diag = true_diagnoses.get(case_id)
                    self._visualize_predictions(
                        viz_dir, case_id, image, patch_info, patch_size, diagnosis, true_diag
                    )

            except Exception as e:
                logger.error(f"Error processing {case_id}: {e}")
                continue

        # Save all predictions
        self._save_diagnosis_predictions(fold_output_dir, all_predictions)

        # Compute metrics if requested
        results = {
            'fold': fold,
            'predictions': all_predictions,
            'patch_info': all_patch_info,
            'num_cases': len(all_predictions)
        }

        if compute_metrics and true_diagnoses:
            metrics = self._compute_metrics(all_predictions, true_diagnoses)
            results['metrics'] = metrics
            self._save_metrics_report(fold_output_dir, metrics, input_dir, labels_dir)

        logger.info(f"Fold {fold} inference complete: {len(all_predictions)} cases processed")

        return results

    def run_inference_ensemble(self, input_dir: Path, output_dir: Path,
                               folds: List[int], batch_size: int = 32,
                               visualize: bool = False,
                               compute_metrics: bool = False,
                               labels_dir: Path = None) -> Dict:
        """
        Run ensemble inference using multiple folds.
        Per-patch majority vote across folds, then diagnosis determination.

        Returns:
            Dictionary with ensemble predictions and optionally metrics
        """
        logger.info(f"Running ensemble inference with folds: {folds}")

        # Create ensemble output directory
        ensemble_dir = output_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        if visualize:
            viz_dir = ensemble_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

        # Find test images
        test_cases = self._find_test_cases(input_dir)
        logger.info(f"Found {len(test_cases)} test cases")

        # Collect predictions from all folds
        fold_predictions = {fold: {} for fold in folds}
        fold_patch_info = {fold: {} for fold in folds}
        patch_sizes = {}

        for fold in folds:
            model, patch_size = self.load_model(fold)
            patch_sizes[fold] = patch_size

            for case_id in test_cases:
                try:
                    image = self.load_test_image(case_id, input_dir)
                    patch_preds, patch_info = self.run_inference_single_image(
                        model, image, patch_size, batch_size
                    )
                    fold_predictions[fold][case_id] = patch_preds
                    fold_patch_info[fold][case_id] = patch_info
                except Exception as e:
                    logger.error(f"Error processing {case_id} with fold {fold}: {e}")

        # Ensemble: per-patch majority vote across folds
        ensemble_predictions = {}
        ensemble_patch_info = {}
        true_diagnoses = {}

        # Use patch_size from first fold (should be same for all)
        patch_size = patch_sizes[folds[0]]

        for case_id in test_cases:
            # Collect patch predictions from all folds for this case
            case_fold_preds = []
            for fold in folds:
                if case_id in fold_predictions[fold]:
                    case_fold_preds.append(fold_predictions[fold][case_id])

            if not case_fold_preds:
                continue

            # Per-patch majority vote
            num_patches = len(case_fold_preds[0])
            ensemble_patch_preds = []

            for patch_idx in range(num_patches):
                patch_votes = [case_fold_preds[f][patch_idx] for f in range(len(case_fold_preds))]
                ensemble_pred = Counter(patch_votes).most_common(1)[0][0]
                ensemble_patch_preds.append(ensemble_pred)

            # Determine ensemble diagnosis
            diagnosis = self.determine_diagnosis(ensemble_patch_preds)
            ensemble_predictions[case_id] = diagnosis

            # Get coordinates from first fold
            if case_id in fold_patch_info[folds[0]]:
                coords = [(p[1], p[2]) for p in fold_patch_info[folds[0]][case_id]]
                ensemble_patch_info[case_id] = [
                    (pred, x, y) for pred, (x, y) in zip(ensemble_patch_preds, coords)
                ]

            # Load ground truth
            if labels_dir and labels_dir.exists():
                label = self.load_test_label(case_id, labels_dir)
                if label is not None:
                    true_diagnoses[case_id] = self.determine_true_diagnosis(label)

            # Save per-case predictions
            if case_id in ensemble_patch_info:
                image = self.load_test_image(case_id, input_dir)
                self._save_case_predictions(
                    ensemble_dir, case_id, ensemble_patch_info[case_id],
                    diagnosis, patch_size, image.shape
                )

                # Create visualization
                if visualize:
                    true_diag = true_diagnoses.get(case_id)
                    self._visualize_predictions(
                        viz_dir, case_id, image, ensemble_patch_info[case_id],
                        patch_size, diagnosis, true_diag
                    )

        # Save all predictions
        self._save_diagnosis_predictions(ensemble_dir, ensemble_predictions)

        # Compute metrics
        results = {
            'folds': folds,
            'predictions': ensemble_predictions,
            'patch_info': ensemble_patch_info,
            'num_cases': len(ensemble_predictions)
        }

        if compute_metrics and true_diagnoses:
            metrics = self._compute_metrics(ensemble_predictions, true_diagnoses)
            results['metrics'] = metrics
            self._save_metrics_report(ensemble_dir, metrics, input_dir, labels_dir)

        logger.info(f"Ensemble inference complete: {len(ensemble_predictions)} cases processed")

        return results

    def _find_test_cases(self, input_dir: Path) -> List[str]:
        """Find all test case IDs in the input directory"""
        case_ids = set()
        for f in input_dir.glob("*_0000.png"):
            case_id = f.stem.rsplit('_', 1)[0]
            case_ids.add(case_id)
        return sorted(case_ids)

    def _save_case_predictions(self, output_dir: Path, case_id: str,
                               patch_info: List[Tuple[int, int, int]],
                               diagnosis: int, patch_size: int,
                               image_shape: Tuple[int, ...]):
        """Save per-case patch predictions to JSON"""
        h, w = image_shape[:2]

        patches_data = []
        for pred, x, y in patch_info:
            # Calculate actual (unpadded) patch dimensions
            actual_w = min(patch_size, w - x)
            actual_h = min(patch_size, h - y)
            is_padded = actual_w < patch_size or actual_h < patch_size

            patches_data.append({
                'x': x, 'y': y,
                'prediction': pred,
                'class_name': CLASS_NAMES[pred],
                'actual_width': actual_w,
                'actual_height': actual_h,
                'is_edge_patch': is_padded
            })

        predictions_data = {
            'case_id': case_id,
            'patch_size': patch_size,
            'image_shape': list(image_shape),
            'patches': patches_data,
            'diagnosis': diagnosis,
            'diagnosis_name': DIAGNOSIS_NAMES[diagnosis]
        }

        output_file = output_dir / f"{case_id}_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

    def _save_diagnosis_predictions(self, output_dir: Path, predictions: Dict[str, int]):
        """Save all diagnosis predictions to JSON"""
        predictions_data = {
            case_id: {
                'diagnosis': diag,
                'diagnosis_name': DIAGNOSIS_NAMES[diag]
            }
            for case_id, diag in predictions.items()
        }

        output_file = output_dir / "diagnosis_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

    def _visualize_predictions(self, viz_dir: Path, case_id: str, image: np.ndarray,
                               patch_info: List[Tuple[int, int, int]], patch_size: int,
                               diagnosis: int, true_diagnosis: int = None):
        """Create grid visualization with color-coded patches"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        h, w = image.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Original image with grid overlay
        axes[0].imshow(image)
        axes[0].set_title('Original Image with Grid', fontsize=12)

        # Draw grid and color each cell (clip to image boundaries for edge patches)
        for pred, x, y in patch_info:
            color = [c/255 for c in COLOR_MAP[pred]]
            # Calculate actual patch dimensions (may be smaller at edges)
            actual_w = min(patch_size, w - x)
            actual_h = min(patch_size, h - y)
            rect = Rectangle((x, y), actual_w, actual_h,
                            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
            axes[0].add_patch(rect)

        axes[0].set_xlim(0, w)
        axes[0].set_ylim(h, 0)
        axes[0].axis('off')

        # Right: Prediction grid (solid colors, clipped to image bounds)
        pred_image = np.zeros((h, w, 3), dtype=np.uint8)
        for pred, x, y in patch_info:
            # Clip to image boundaries for edge patches
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            pred_image[y:y_end, x:x_end] = COLOR_MAP[pred]

        axes[1].imshow(pred_image)

        # Title with diagnosis
        title = f'Predicted: {DIAGNOSIS_NAMES[diagnosis]}'
        if true_diagnosis is not None:
            title += f' | True: {DIAGNOSIS_NAMES[true_diagnosis]}'
            correct = diagnosis == true_diagnosis
            title += f' | {"CORRECT" if correct else "INCORRECT"}'
        axes[1].set_title(title, fontsize=12)
        axes[1].axis('off')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=[c/255 for c in COLOR_MAP[i]],
                      markersize=15, label=DIAGNOSIS_NAMES[i])
            for i in range(4)
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        output_file = viz_dir / f"{case_id}_viz.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

    def _compute_metrics(self, predictions: Dict[str, int],
                         true_diagnoses: Dict[str, int]) -> Dict:
        """Compute diagnosis-level classification metrics"""
        # Filter to cases with both prediction and ground truth
        common_cases = set(predictions.keys()) & set(true_diagnoses.keys())

        y_pred = [predictions[case_id] for case_id in common_cases]
        y_true = [true_diagnoses[case_id] for case_id in common_cases]

        metrics = {
            'num_cases': len(common_cases),
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

        metrics['per_diagnosis'] = {}
        for i, diag_name in enumerate(DIAGNOSIS_NAMES):
            support = sum(1 for t in y_true if t == i)
            metrics['per_diagnosis'][diag_name] = {
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
            y_true, y_pred, target_names=DIAGNOSIS_NAMES, labels=[0, 1, 2, 3], zero_division=0
        )

        return metrics

    def _save_metrics_report(self, output_dir: Path, metrics: Dict,
                             input_dir: Path, labels_dir: Path):
        """Save metrics report to text and JSON files"""
        # Save JSON
        json_file = output_dir / "diagnosis_report.json"
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save text report
        txt_file = output_dir / "diagnosis_report.txt"
        with open(txt_file, 'w') as f:
            f.write("DIAGNOSIS CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Model: {self.model_name} ({self.model_full_name})\n")
            f.write(f"Dataset: {self.dataset_full_name}\n")
            f.write(f"Input directory: {input_dir}\n")
            f.write(f"Labels directory: {labels_dir}\n")
            f.write(f"Total cases: {metrics['num_cases']}\n")
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
            for name in DIAGNOSIS_NAMES:
                f.write(f"{name:>12s}")
            f.write("\n" + "-"*63 + "\n")

            for i, name in enumerate(DIAGNOSIS_NAMES):
                f.write(f"{name:15s}")
                for j in range(4):
                    f.write(f"{cm[i, j]:12d}")
                f.write("\n")
            f.write("\n")

            f.write("PER-DIAGNOSIS METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Diagnosis':<15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}\n")
            f.write("-"*63 + "\n")

            for diag_name in DIAGNOSIS_NAMES:
                m = metrics['per_diagnosis'][diag_name]
                f.write(f"{diag_name:<15s} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                       f"{m['f1']:>10.4f} {m['support']:>10d}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("SKLEARN CLASSIFICATION REPORT:\n")
            f.write("-"*80 + "\n")
            f.write(metrics['classification_report'])

        logger.info(f"Metrics report saved: {txt_file}")
        logger.info(f"JSON metrics saved: {json_file}")

    def create_metrics_summary(self, output_dir: Path, fold_results: Dict[int, Dict],
                               ensemble_results: Dict = None):
        """Create summary CSV comparing metrics across folds"""
        import csv

        summary_file = output_dir / "metrics_summary.csv"

        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['Fold', 'Accuracy', 'Balanced_Accuracy', 'Kappa', 'MCC', 'Macro_F1', 'Weighted_F1']
            for diag in DIAGNOSIS_NAMES:
                header.extend([f'{diag}_Precision', f'{diag}_Recall', f'{diag}_F1'])
            writer.writerow(header)

            # Write fold results
            for fold, results in sorted(fold_results.items()):
                if 'metrics' not in results:
                    continue
                m = results['metrics']
                row = [
                    f'fold_{fold}',
                    f"{m['accuracy']:.4f}",
                    f"{m['balanced_accuracy']:.4f}",
                    f"{m['kappa']:.4f}",
                    f"{m['mcc']:.4f}",
                    f"{m['macro_f1']:.4f}",
                    f"{m['weighted_f1']:.4f}",
                ]
                for diag in DIAGNOSIS_NAMES:
                    pd = m['per_diagnosis'][diag]
                    row.extend([f"{pd['precision']:.4f}", f"{pd['recall']:.4f}", f"{pd['f1']:.4f}"])
                writer.writerow(row)

            # Write ensemble results
            if ensemble_results and 'metrics' in ensemble_results:
                m = ensemble_results['metrics']
                row = [
                    'ensemble',
                    f"{m['accuracy']:.4f}",
                    f"{m['balanced_accuracy']:.4f}",
                    f"{m['kappa']:.4f}",
                    f"{m['mcc']:.4f}",
                    f"{m['macro_f1']:.4f}",
                    f"{m['weighted_f1']:.4f}",
                ]
                for diag in DIAGNOSIS_NAMES:
                    pd = m['per_diagnosis'][diag]
                    row.extend([f"{pd['precision']:.4f}", f"{pd['recall']:.4f}", f"{pd['f1']:.4f}"])
                writer.writerow(row)

            # Compute and write mean/std across folds
            if len(fold_results) > 1:
                metrics_keys = ['accuracy', 'balanced_accuracy', 'kappa', 'mcc', 'macro_f1', 'weighted_f1']
                fold_metrics = {k: [] for k in metrics_keys}

                for fold, results in fold_results.items():
                    if 'metrics' not in results:
                        continue
                    for k in metrics_keys:
                        fold_metrics[k].append(results['metrics'][k])

                # Mean row
                mean_row = ['Mean']
                for k in metrics_keys:
                    mean_row.append(f"{np.mean(fold_metrics[k]):.4f}")
                # Per-diagnosis means
                for diag in DIAGNOSIS_NAMES:
                    for metric in ['precision', 'recall', 'f1']:
                        values = [r['metrics']['per_diagnosis'][diag][metric]
                                 for r in fold_results.values() if 'metrics' in r]
                        mean_row.append(f"{np.mean(values):.4f}")
                writer.writerow(mean_row)

                # Std row
                std_row = ['Std']
                for k in metrics_keys:
                    std_row.append(f"{np.std(fold_metrics[k]):.4f}")
                for diag in DIAGNOSIS_NAMES:
                    for metric in ['precision', 'recall', 'f1']:
                        values = [r['metrics']['per_diagnosis'][diag][metric]
                                 for r in fold_results.values() if 'metrics' in r]
                        std_row.append(f"{np.std(values):.4f}")
                writer.writerow(std_row)

        logger.info(f"Metrics summary saved: {summary_file}")

    def create_inference_summary(self, output_dir: Path, folds: List[int],
                                 inference_time: float = None):
        """Create summary of inference run"""
        summary = {
            "dataset": {
                "id": self.dataset_id,
                "name": self.dataset_name,
                "full_name": self.dataset_full_name
            },
            "model": {
                "name": self.model_name,
                "full_name": self.model_full_name
            },
            "inference": {
                "folds_used": folds,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.device)
            }
        }

        if inference_time:
            summary["inference"]["time_seconds"] = inference_time

        summary_file = output_dir / "inference_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Inference summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Classification Inference Script for Test Set')

    # Required arguments
    parser.add_argument('--dataset_id', type=int, required=True,
                       help='Dataset ID (100-104)')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODELS.keys()),
                       help='Model name')

    # Optional arguments
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name (auto-detected if not provided)')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing test images (default: nnUNet_raw/.../imagesTs)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save predictions (default: predictions_classification/...)')

    # Fold selection
    parser.add_argument('--fold', type=int,
                       help='Use specific fold for inference')
    parser.add_argument('--folds', nargs='+', type=int,
                       help='Use specific folds for ensemble (e.g., --folds 0 1 2)')
    parser.add_argument('--all_folds', action='store_true',
                       help='Use all available trained folds')

    # Output options
    parser.add_argument('--visualize', action='store_true',
                       help='Create grid visualizations of predictions')
    parser.add_argument('--compute_metrics', action='store_true',
                       help='Compute and save classification metrics')
    parser.add_argument('--generate_summary', action='store_true',
                       help='Generate summary CSV across folds')
    parser.add_argument('--labels_dir', type=str,
                       help='Directory containing ground truth labels (for metrics)')

    # Device options
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')

    # Additional options
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (default: 32)')

    args = parser.parse_args()

    # Auto-detect dataset name
    if args.dataset_name is None:
        if args.dataset_id in DATASETS:
            args.dataset_name = DATASETS[args.dataset_id]
        else:
            logger.error(f"Unknown dataset ID {args.dataset_id}, please provide --dataset_name")
            return 1

    dataset_full_name = f"Dataset{args.dataset_id}_{args.dataset_name}"

    # Setup paths
    if args.input_dir is None:
        args.input_dir = Path("nnUNet_raw") / dataset_full_name / "imagesTs"
    else:
        args.input_dir = Path(args.input_dir)

    if args.output_dir is None:
        args.output_dir = Path("predictions_classification") / dataset_full_name / args.model
    else:
        args.output_dir = Path(args.output_dir)

    if args.labels_dir:
        args.labels_dir = Path(args.labels_dir)
    elif args.compute_metrics:
        # Default labels directory
        args.labels_dir = Path("nnUNet_raw") / dataset_full_name / "labelsTs"

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    # Create inference manager
    inference = ClassificationInference(args.dataset_id, args.dataset_name, args.model)
    inference.set_device(use_cpu=args.cpu, gpu_id=args.gpu)

    # Get available folds
    available_folds = inference.get_trained_folds()
    if not available_folds:
        logger.error(f"No trained folds found for model {args.model}")
        return 1

    # Determine which folds to use
    if args.all_folds:
        folds = available_folds
    elif args.folds:
        folds = [f for f in args.folds if f in available_folds]
        if len(folds) != len(args.folds):
            missing = set(args.folds) - set(folds)
            logger.warning(f"Folds not found: {missing}")
    elif args.fold is not None:
        if args.fold not in available_folds:
            logger.error(f"Fold {args.fold} not found. Available: {available_folds}")
            return 1
        folds = [args.fold]
    else:
        # Default to all folds
        folds = available_folds

    logger.info(f"Using folds: {folds}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Start inference
    start_time = datetime.now()
    fold_results = {}

    # Run inference on each fold
    for fold in folds:
        results = inference.run_inference_single_fold(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            fold=fold,
            batch_size=args.batch_size,
            visualize=args.visualize,
            compute_metrics=args.compute_metrics,
            labels_dir=args.labels_dir
        )
        fold_results[fold] = results

    # Run ensemble if multiple folds
    ensemble_results = None
    if len(folds) > 1:
        ensemble_results = inference.run_inference_ensemble(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            folds=folds,
            batch_size=args.batch_size,
            visualize=args.visualize,
            compute_metrics=args.compute_metrics,
            labels_dir=args.labels_dir
        )

    # Generate summary
    if args.generate_summary and args.compute_metrics:
        inference.create_metrics_summary(args.output_dir, fold_results, ensemble_results)

    # Save inference summary
    inference_time = (datetime.now() - start_time).total_seconds()
    inference.create_inference_summary(args.output_dir, folds, inference_time)

    logger.info("="*80)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Total time: {inference_time:.1f} seconds")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
