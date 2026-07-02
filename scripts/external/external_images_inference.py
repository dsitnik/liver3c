#!/usr/bin/env python3
"""
External Images Inference Script
Runs inference on external JPG images using ALL trained model ensembles across ALL datasets.

Supports:
- nnUNet (5 datasets × 10-fold ensemble)
- Classification models: convnextv2, densenet, efficientnetv2, maxvit, swinv2 (5 models × 5 datasets)
- Total: 30 ensemble inference runs

Usage:
    # Full run (all models, all datasets)
    python external_images_inference.py

    # Single model + single dataset
    python external_images_inference.py --models convnextv2 --datasets 104

    # nnUNet only
    python external_images_inference.py --models nnunet --datasets 104
"""

import os
import json
import argparse
import subprocess
import sys
import io
import logging
import shutil
import time
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

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('external_images_inference.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
CLASSIFICATION_MODELS = {
    'convnextv2': 'convnextv2_base.fcmae_ft_in22k_in1k',
    'efficientnetv2': 'tf_efficientnetv2_m.in21k_ft_in1k',
    'swinv2': 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
    'maxvit': 'maxvit_base_tf_384.in21k_ft_in1k',
    'densenet': 'densenet161.tv_in1k'
}

DATASETS = {
    100: 'Liver1',
    101: 'Liver2',
    102: 'Liver3',
    103: 'Liver4',
    104: 'Liver5'
}

CLASS_NAMES = {0: 'background', 1: 'metastatic', 2: 'hcc', 3: 'cho'}
DIAGNOSIS_NAMES = ['Healthy', 'Metastatic', 'HCC', 'CHO']

COLOR_MAP = {
    0: [0, 0, 0],       # Background - Black
    1: [255, 0, 0],     # Metastatic - Red
    2: [0, 255, 0],     # HCC - Green
    3: [0, 0, 255]      # CHO - Blue
}


class ImagePreparer:
    """Converts external JPG images to 3-channel PNG format for inference."""

    def __init__(self, input_dir: Path, prepared_dir: Path):
        self.input_dir = input_dir
        self.prepared_dir = prepared_dir

    def prepare(self) -> List[str]:
        """
        Load each JPG, split RGB into 3 grayscale PNGs.
        No normalization or resizing - models handle this internally.

        Returns:
            List of case IDs
        """
        logger.info("=" * 60)
        logger.info("Preparing external images")
        logger.info(f"  Input: {self.input_dir}")
        logger.info(f"  Output: {self.prepared_dir}")
        logger.info("=" * 60)

        self.prepared_dir.mkdir(parents=True, exist_ok=True)

        # Use a set to avoid duplicates on case-insensitive filesystems (Windows)
        image_files = sorted({f for f in self.input_dir.glob("*.jpg")} | {f for f in self.input_dir.glob("*.JPG")})
        if not image_files:
            raise ValueError(f"No JPG images found in {self.input_dir}")

        logger.info(f"Found {len(image_files)} images to process")

        case_ids = []
        for img_file in image_files:
            case_id = img_file.stem
            img = Image.open(img_file)
            img_array = np.array(img)

            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                logger.warning(f"Skipping {img_file.name}: not a 3-channel RGB image")
                continue

            # Split into channels and save as PNGs (no normalization)
            for ch in range(3):
                channel_data = img_array[:, :, ch]
                if channel_data.max() <= 1.0:
                    channel_data = (channel_data * 255).astype(np.uint8)
                else:
                    channel_data = np.clip(channel_data, 0, 255).astype(np.uint8)

                output_file = self.prepared_dir / f"{case_id}_{ch:04d}.png"
                Image.fromarray(channel_data).save(output_file)

            case_ids.append(case_id)

        logger.info(f"Prepared {len(case_ids)} images")
        return case_ids


class NNUNetExternalInference:
    """Runs nnUNet inference on external images for a specific dataset."""

    def __init__(self, dataset_id: int, dataset_name: str):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.dataset_full_name = f"Dataset{dataset_id}_{dataset_name}"

        current_dir = Path.cwd()
        self.nnunet_raw = current_dir / "nnUNet_raw"
        self.nnunet_preprocessed = current_dir / "nnUNet_preprocessed"
        self.nnunet_results = current_dir / "nnUNet_results"
        self.results_path = self.nnunet_results / self.dataset_full_name

        # Set environment variables
        os.environ['nnUNet_raw'] = str(self.nnunet_raw)
        os.environ['nnUNet_preprocessed'] = str(self.nnunet_preprocessed)
        os.environ['nnUNet_results'] = str(self.nnunet_results)

    def get_trained_folds(self, configuration: str = "2d",
                          trainer: str = "nnUNetTrainer_500epochs") -> List[int]:
        """Auto-detect trained folds from checkpoint files."""
        trained_folds = []
        config_path = self.results_path / f"{trainer}__nnUNetPlans__{configuration}"

        if not config_path.exists():
            logger.warning(f"Config path not found: {config_path}")
            return trained_folds

        for fold_dir in config_path.iterdir():
            if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
                fold_num = int(fold_dir.name.split("_")[1])
                if (fold_dir / "checkpoint_best.pth").exists():
                    trained_folds.append(fold_num)

        trained_folds.sort()
        return trained_folds

    def run_ensemble(self, prepared_dir: Path, output_dir: Path,
                     configuration: str = "2d",
                     trainer: str = "nnUNetTrainer_500epochs",
                     use_gpu: bool = True, gpu_id: int = 0) -> bool:
        """Run ensemble inference using all available folds."""
        folds = self.get_trained_folds(configuration, trainer)
        if not folds:
            logger.error(f"No trained folds found for {self.dataset_full_name}")
            return False

        logger.info(f"nnUNet ensemble for {self.dataset_full_name}: folds {folds}")

        ensemble_dir = output_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        nnunet_predict = str(Path(sys.executable).parent / "nnUNetv2_predict")
        cmd = [
            nnunet_predict,
            "-i", str(prepared_dir),
            "-o", str(ensemble_dir),
            "-d", str(self.dataset_id),
            "-c", configuration,
            "-f"
        ] + [str(f) for f in folds] + [
            "-tr", trainer,
            "-chk", "checkpoint_best.pth",
            "-step_size", "0.5"
        ]

        if not use_gpu:
            cmd.extend(["-device", "cpu"])
        elif gpu_id != 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"nnUNet inference failed for {self.dataset_full_name}: {e}")
            return False

        return True

    def generate_diagnosis(self, predictions_dir: Path) -> Dict[str, dict]:
        """Generate diagnosis from prediction masks using majority tumor pixel class."""
        diagnosis_results = {}

        for pred_file in sorted(predictions_dir.glob("*.png")):
            pred_array = np.array(Image.open(pred_file))
            case_id = pred_file.stem

            # Count pixels per class
            unique, counts = np.unique(pred_array, return_counts=True)
            pixel_counts = dict(zip(unique.tolist(), counts.tolist()))

            # Determine diagnosis
            tumor_counts = {c: pixel_counts.get(c, 0) for c in [1, 2, 3]}
            total_tumor = sum(tumor_counts.values())

            if total_tumor == 0:
                diagnosis = 0  # Healthy
            else:
                diagnosis = max(tumor_counts, key=tumor_counts.get)

            diagnosis_results[case_id] = {
                'diagnosis': diagnosis,
                'diagnosis_name': DIAGNOSIS_NAMES[diagnosis],
                'pixel_counts': pixel_counts,
                'total_pixels': int(pred_array.size)
            }

        return diagnosis_results

    def create_visualizations(self, predictions_dir: Path, original_images_dir: Path,
                              viz_dir: Path):
        """Create color-coded masks and 50% overlays with original images."""
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Build case_id -> original file mapping
        original_map = {}
        for f in sorted({f for f in original_images_dir.glob("*.jpg")} | {f for f in original_images_dir.glob("*.JPG")}):
            original_map[f.stem] = f

        for pred_file in sorted(predictions_dir.glob("*.png")):
            pred_array = np.array(Image.open(pred_file))
            case_id = pred_file.stem
            h, w = pred_array.shape[:2]

            # Color-coded mask
            rgb_viz = np.zeros((h, w, 3), dtype=np.uint8)
            for class_idx, color in COLOR_MAP.items():
                mask = pred_array == class_idx
                rgb_viz[mask] = color

            Image.fromarray(rgb_viz).save(viz_dir / f"{case_id}_colored.png")

            # Overlay with original
            if case_id in original_map:
                orig_img = Image.open(original_map[case_id])
                orig_array = np.array(orig_img)

                # Resize original to match prediction if needed
                if orig_array.shape[:2] != (h, w):
                    orig_img = orig_img.resize((w, h), Image.BICUBIC)
                    orig_array = np.array(orig_img)

                overlay = (0.5 * orig_array + 0.5 * rgb_viz).astype(np.uint8)
                Image.fromarray(overlay).save(viz_dir / f"{case_id}_overlay.png")

        logger.info(f"Visualizations saved to {viz_dir}")


class PatchDataset(Dataset):
    """Dataset for batch inference on extracted patches."""

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
            patch = patch.astype(np.float32) / 255.0
            if self.normalization_stats:
                mean, std = self.normalization_stats
                for c in range(3):
                    patch[:, :, c] = (patch[:, :, c] - mean[c]) / std[c]
            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1))

        return patch_tensor


class ClassificationExternalInference:
    """Runs classification inference on external images for a specific model + dataset."""

    def __init__(self, model_name: str, dataset_id: int, dataset_name: str):
        self.model_name = model_name
        self.model_full_name = CLASSIFICATION_MODELS[model_name]
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.dataset_full_name = f"Dataset{dataset_id}_{dataset_name}"

        current_dir = Path.cwd()
        self.preprocessed_path = current_dir / "nnUNet_preprocessed" / self.dataset_full_name
        self.results_path = current_dir / "classification_results" / self.dataset_full_name / model_name

        # Load normalization stats
        self.mean, self.std = self._load_normalization_stats()
        self.transform = self._get_inference_transforms()

        self.device = torch.device('cpu')

    def set_device(self, use_cpu: bool = False, gpu_id: int = 0):
        if use_cpu:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')

    def _load_normalization_stats(self) -> Tuple[List[float], List[float]]:
        fingerprint_path = self.preprocessed_path / "dataset_fingerprint.json"

        if not fingerprint_path.exists():
            logger.warning(f"Fingerprint not found: {fingerprint_path}, using defaults")
            return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        with open(fingerprint_path) as f:
            fingerprint = json.load(f)

        intensity_props = fingerprint.get('foreground_intensity_properties_per_channel', {})

        means, stds = [], []
        for ch in range(3):
            ch_props = intensity_props.get(str(ch), {})
            means.append(ch_props.get('mean', 127.5) / 255.0)
            stds.append(ch_props.get('std', 63.75) / 255.0)

        return means, stds

    def _get_inference_transforms(self):
        if not ALBUMENTATIONS_AVAILABLE:
            return None
        return A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def get_trained_folds(self) -> List[int]:
        trained_folds = []
        if self.results_path.exists():
            for fold_dir in self.results_path.iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
                    fold_num = int(fold_dir.name.split("_")[1])
                    if (fold_dir / "checkpoint_best.pth").exists():
                        trained_folds.append(fold_num)
        trained_folds.sort()
        return trained_folds

    def load_model(self, fold: int) -> Tuple[nn.Module, int]:
        fold_path = self.results_path / f"fold_{fold}"
        checkpoint_path = fold_path / "checkpoint_best.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        config = checkpoint.get('config', {})
        patch_size = config.get('patch_size', checkpoint.get('patch_size', 384))

        model = timm.create_model(
            self.model_full_name,
            pretrained=False,
            num_classes=4,
            in_chans=3,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        logger.info(f"Loaded {self.model_name} fold {fold}, patch_size={patch_size}")
        return model, patch_size

    def load_image(self, case_id: str, prepared_dir: Path) -> np.ndarray:
        """Load 3-channel image from prepared PNGs."""
        channels = []
        for ch in range(3):
            channel_path = prepared_dir / f"{case_id}_{ch:04d}.png"
            if not channel_path.exists():
                raise FileNotFoundError(f"Channel file not found: {channel_path}")
            channels.append(np.array(Image.open(channel_path)))
        return np.stack(channels, axis=-1)

    def extract_patches(self, image: np.ndarray, patch_size: int) -> List[Tuple[np.ndarray, int, int]]:
        """Extract non-overlapping patches with zero-padding at edges."""
        h, w = image.shape[:2]
        patches = []

        n_patches_y = (h + patch_size - 1) // patch_size
        n_patches_x = (w + patch_size - 1) // patch_size

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y = i * patch_size
                x = j * patch_size
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                patch = image[y:y_end, x:x_end, :]

                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    padded[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded

                patches.append((patch, x, y))

        return patches

    def determine_diagnosis(self, patch_predictions: List[int]) -> int:
        if not patch_predictions or all(p == 0 for p in patch_predictions):
            return 0
        tumor_preds = [p for p in patch_predictions if p > 0]
        if not tumor_preds:
            return 0
        return Counter(tumor_preds).most_common(1)[0][0]

    def run_inference_single_image(self, model: nn.Module, image: np.ndarray,
                                   patch_size: int, batch_size: int = 32) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        patches_with_coords = self.extract_patches(image, patch_size)
        if not patches_with_coords:
            return [], []

        patches = [p[0] for p in patches_with_coords]
        coords = [(p[1], p[2]) for p in patches_with_coords]

        dataset = PatchDataset(patches, self.transform, (self.mean, self.std))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                outputs = model(batch)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                all_predictions.extend(predictions.tolist())

        patch_info = [(pred, x, y) for pred, (x, y) in zip(all_predictions, coords)]
        return all_predictions, patch_info

    def run_ensemble(self, prepared_dir: Path, output_dir: Path,
                     case_ids: List[str], batch_size: int = 32,
                     visualize: bool = True, original_images_dir: Path = None) -> bool:
        """Run ensemble inference across all folds."""
        folds = self.get_trained_folds()
        if not folds:
            logger.error(f"No trained folds for {self.model_name}/{self.dataset_full_name}")
            return False

        logger.info(f"Classification ensemble: {self.model_name}/{self.dataset_full_name}, folds {folds}")

        ensemble_dir = output_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        if visualize:
            viz_dir = ensemble_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

        # Collect per-fold predictions for all cases
        fold_predictions = {fold: {} for fold in folds}
        fold_patch_info = {fold: {} for fold in folds}
        patch_sizes = {}

        for fold in folds:
            try:
                model, patch_size = self.load_model(fold)
                patch_sizes[fold] = patch_size
            except Exception as e:
                logger.error(f"Failed to load model fold {fold}: {e}")
                continue

            for case_id in case_ids:
                try:
                    image = self.load_image(case_id, prepared_dir)
                    patch_preds, patch_info = self.run_inference_single_image(
                        model, image, patch_size, batch_size
                    )
                    fold_predictions[fold][case_id] = patch_preds
                    fold_patch_info[fold][case_id] = patch_info
                except Exception as e:
                    logger.error(f"Error processing {case_id} with fold {fold}: {e}")

            # Free GPU memory between folds
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Per-patch majority vote across folds
        active_folds = [f for f in folds if f in fold_predictions and fold_predictions[f]]
        if not active_folds:
            logger.error("No successful fold predictions")
            return False

        patch_size = patch_sizes[active_folds[0]]
        ensemble_predictions = {}

        for case_id in case_ids:
            case_fold_preds = []
            for fold in active_folds:
                if case_id in fold_predictions[fold]:
                    case_fold_preds.append(fold_predictions[fold][case_id])

            if not case_fold_preds:
                continue

            num_patches = len(case_fold_preds[0])
            ensemble_patch_preds = []
            for patch_idx in range(num_patches):
                patch_votes = [cfp[patch_idx] for cfp in case_fold_preds]
                ensemble_patch_preds.append(Counter(patch_votes).most_common(1)[0][0])

            diagnosis = self.determine_diagnosis(ensemble_patch_preds)

            # Get coords from first active fold
            first_fold = active_folds[0]
            if case_id in fold_patch_info[first_fold]:
                coords = [(p[1], p[2]) for p in fold_patch_info[first_fold][case_id]]
                ensemble_patch_info = [
                    (pred, x, y) for pred, (x, y) in zip(ensemble_patch_preds, coords)
                ]
            else:
                ensemble_patch_info = []

            ensemble_predictions[case_id] = {
                'diagnosis': diagnosis,
                'diagnosis_name': DIAGNOSIS_NAMES[diagnosis],
                'patch_predictions': ensemble_patch_preds,
                'patch_info': [(p, x, y) for p, x, y in ensemble_patch_info]
            }

            # Save per-case JSON
            image = self.load_image(case_id, prepared_dir)
            h, w = image.shape[:2]
            self._save_case_predictions(
                ensemble_dir, case_id, ensemble_patch_info,
                diagnosis, patch_size, image.shape
            )

            # Visualization
            if visualize and original_images_dir:
                self._visualize_predictions(
                    viz_dir, case_id, image, ensemble_patch_info,
                    patch_size, diagnosis, original_images_dir
                )

        # Save diagnosis_predictions.json
        diag_data = {
            case_id: {
                'diagnosis': info['diagnosis'],
                'diagnosis_name': info['diagnosis_name']
            }
            for case_id, info in ensemble_predictions.items()
        }
        with open(ensemble_dir / "diagnosis_predictions.json", 'w') as f:
            json.dump(diag_data, f, indent=2)

        logger.info(f"Ensemble complete: {len(ensemble_predictions)} cases")
        return True

    def _save_case_predictions(self, output_dir: Path, case_id: str,
                               patch_info: List[Tuple[int, int, int]],
                               diagnosis: int, patch_size: int,
                               image_shape: Tuple[int, ...]):
        h, w = image_shape[:2]
        patches_data = []
        for pred, x, y in patch_info:
            actual_w = min(patch_size, w - x)
            actual_h = min(patch_size, h - y)
            patches_data.append({
                'x': x, 'y': y,
                'prediction': pred,
                'class_name': CLASS_NAMES[pred],
                'actual_width': actual_w,
                'actual_height': actual_h,
                'is_edge_patch': actual_w < patch_size or actual_h < patch_size
            })

        predictions_data = {
            'case_id': case_id,
            'patch_size': patch_size,
            'image_shape': list(image_shape),
            'patches': patches_data,
            'diagnosis': diagnosis,
            'diagnosis_name': DIAGNOSIS_NAMES[diagnosis]
        }

        with open(output_dir / f"{case_id}_predictions.json", 'w') as f:
            json.dump(predictions_data, f, indent=2)

    def _visualize_predictions(self, viz_dir: Path, case_id: str, image: np.ndarray,
                               patch_info: List[Tuple[int, int, int]], patch_size: int,
                               diagnosis: int, original_images_dir: Path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        h, w = image.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Original with grid overlay
        axes[0].imshow(image)
        axes[0].set_title('Original Image with Grid', fontsize=12)

        for pred, x, y in patch_info:
            color = [c / 255 for c in COLOR_MAP[pred]]
            actual_w = min(patch_size, w - x)
            actual_h = min(patch_size, h - y)
            rect = Rectangle((x, y), actual_w, actual_h,
                              linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
            axes[0].add_patch(rect)

        axes[0].set_xlim(0, w)
        axes[0].set_ylim(h, 0)
        axes[0].axis('off')

        # Right: Prediction grid
        pred_image = np.zeros((h, w, 3), dtype=np.uint8)
        for pred, x, y in patch_info:
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            pred_image[y:y_end, x:x_end] = COLOR_MAP[pred]

        axes[1].imshow(pred_image)
        axes[1].set_title(f'Predicted: {DIAGNOSIS_NAMES[diagnosis]}', fontsize=12)
        axes[1].axis('off')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=[c / 255 for c in COLOR_MAP[i]],
                       markersize=15, label=DIAGNOSIS_NAMES[i])
            for i in range(4)
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(viz_dir / f"{case_id}_viz.png", dpi=150, bbox_inches='tight')
        plt.close()


class ExternalImagesInference:
    """Main orchestrator for running all inference runs."""

    def __init__(self, args):
        self.args = args
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.prepared_dir = Path("_external_images_prepared")

    def run(self):
        start_time = time.time()

        # Step 1: Prepare images (raw uint8 channel PNGs, no normalization)
        # This matches how images are stored in nnUNet_raw/ (the format both
        # nnUNet and classification models were trained on).
        # - nnUNet applies per-image ZScoreNormalization internally during predict
        # - Classification models apply normalization via albumentations transform
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Preparing external images (raw uint8 channel PNGs)")
        logger.info("=" * 80)

        preparer = ImagePreparer(self.input_dir, self.prepared_dir)
        case_ids = preparer.prepare()

        if not case_ids:
            logger.error("No images were prepared")
            return

        # Determine which models to run
        models_to_run = self.args.models
        datasets_to_run = self.args.datasets

        total_runs = 0
        completed_runs = 0
        failed_runs = []

        for model in models_to_run:
            for dataset_id in datasets_to_run:
                total_runs += 1

        logger.info(f"\nTotal inference runs planned: {total_runs}")
        logger.info(f"Models: {models_to_run}")
        logger.info(f"Datasets: {datasets_to_run}")

        # Step 2: Run nnUNet inference
        # nnUNetv2_predict handles preprocessing internally (ZScoreNormalization
        # per image, resampling, sliding window) — no manual preprocessing needed.
        if 'nnunet' in models_to_run:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: nnUNet Inference")
            logger.info("=" * 80)

            for dataset_id in datasets_to_run:
                dataset_name = DATASETS.get(dataset_id)
                if not dataset_name:
                    logger.warning(f"Unknown dataset ID: {dataset_id}")
                    failed_runs.append(f"nnunet/Dataset{dataset_id}")
                    continue

                dataset_label = f"Dataset{dataset_id}_{dataset_name}"
                run_output = self.output_dir / "nnunet" / dataset_label

                logger.info(f"\n--- nnUNet: {dataset_label} ---")

                nnunet = NNUNetExternalInference(dataset_id, dataset_name)
                success = nnunet.run_ensemble(
                    self.prepared_dir, run_output,
                    use_gpu=not self.args.cpu,
                    gpu_id=self.args.gpu
                )

                if success:
                    ensemble_dir = run_output / "ensemble"

                    # Generate diagnosis
                    diagnosis_results = nnunet.generate_diagnosis(ensemble_dir)
                    with open(run_output / "diagnosis_predictions.json", 'w') as f:
                        json.dump(diagnosis_results, f, indent=2)
                    logger.info(f"Saved diagnosis predictions for {len(diagnosis_results)} cases")

                    # Visualizations
                    if not self.args.skip_visualization:
                        viz_dir = ensemble_dir / "visualizations"
                        nnunet.create_visualizations(ensemble_dir, self.input_dir, viz_dir)

                    completed_runs += 1
                else:
                    failed_runs.append(f"nnunet/{dataset_label}")

        # Step 3: Classification inference
        # Classification models apply dataset-specific normalization via
        # albumentations A.Normalize(mean, std) — same as during training.
        classification_models = [m for m in models_to_run if m != 'nnunet']

        if classification_models:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Classification Inference")
            logger.info("=" * 80)

            for model_name in classification_models:
                for dataset_id in datasets_to_run:
                    dataset_name = DATASETS.get(dataset_id)
                    if not dataset_name:
                        logger.warning(f"Unknown dataset ID: {dataset_id}")
                        failed_runs.append(f"{model_name}/Dataset{dataset_id}")
                        continue

                    dataset_label = f"Dataset{dataset_id}_{dataset_name}"
                    run_output = self.output_dir / model_name / dataset_label

                    logger.info(f"\n--- {model_name}: {dataset_label} ---")

                    try:
                        clf = ClassificationExternalInference(
                            model_name, dataset_id, dataset_name
                        )
                        clf.set_device(use_cpu=self.args.cpu, gpu_id=self.args.gpu)

                        success = clf.run_ensemble(
                            self.prepared_dir, run_output,
                            case_ids=case_ids,
                            batch_size=self.args.batch_size,
                            visualize=not self.args.skip_visualization,
                            original_images_dir=self.input_dir
                        )

                        if success:
                            completed_runs += 1
                        else:
                            failed_runs.append(f"{model_name}/{dataset_label}")
                    except Exception as e:
                        logger.error(f"Failed {model_name}/{dataset_label}: {e}")
                        import traceback
                        traceback.print_exc()
                        failed_runs.append(f"{model_name}/{dataset_label}")

        # Step 4: Cleanup
        if not self.args.keep_prepared and self.prepared_dir.exists():
            shutil.rmtree(self.prepared_dir, ignore_errors=True)
            logger.info("Cleaned up prepared images directory")

        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("INFERENCE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"Completed: {completed_runs}/{total_runs} runs")
        if failed_runs:
            logger.info(f"Failed: {failed_runs}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='External Images Inference - All models, all datasets'
    )

    parser.add_argument('--input_dir', type=str, default='data/external_images',
                        help='Directory containing external JPG images (default: data/external_images)')
    parser.add_argument('--output_dir', type=str, default='predictions_external',
                        help='Directory to save predictions (default: predictions_external)')
    parser.add_argument('--models', nargs='+',
                        choices=['nnunet', 'convnextv2', 'densenet', 'efficientnetv2', 'maxvit', 'swinv2'],
                        default=['nnunet', 'convnextv2', 'densenet', 'efficientnetv2', 'maxvit', 'swinv2'],
                        help='Models to run (default: all)')
    parser.add_argument('--datasets', nargs='+', type=int,
                        default=[100, 101, 102, 103, 104],
                        help='Dataset IDs (default: 100 101 102 103 104)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for classification inference (default: 32)')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip creating visualizations')
    parser.add_argument('--keep_prepared', action='store_true',
                        help='Keep prepared images (do not delete temp directory)')

    args = parser.parse_args()

    # Validate input directory
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    orchestrator = ExternalImagesInference(args)
    orchestrator.run()


if __name__ == "__main__":
    main()
