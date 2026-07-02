#!/usr/bin/env python3
"""
nnUNet Patch Classification Script

Converts nnUNet pixel-level prediction PNGs into patch-based classification format
matching classification model outputs. Extracts non-overlapping patches from the
segmentation mask, assigns each patch a class via majority pixel vote, then determines
image-level diagnosis via majority tumor patch vote.

Usage:
    python nnunet_patch_classification.py
    python nnunet_patch_classification.py --datasets 104
    python nnunet_patch_classification.py --skip_visualization
"""

import os
import sys
import io
import json
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nnunet_patch_classification.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
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


def extract_patch_grid(h: int, w: int, patch_size: int) -> List[Tuple[int, int]]:
    """Get top-left coordinates for non-overlapping patch grid (including edge patches)."""
    n_patches_y = (h + patch_size - 1) // patch_size
    n_patches_x = (w + patch_size - 1) // patch_size
    coords = []
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            coords.append((j * patch_size, i * patch_size))
    return coords


def classify_patch(mask_patch: np.ndarray) -> int:
    """Classify a patch by majority pixel vote among non-zero (tumor) pixels."""
    unique, counts = np.unique(mask_patch, return_counts=True)
    pixel_counts = dict(zip(unique.tolist(), counts.tolist()))

    # Remove background count for tumor voting
    tumor_counts = {k: v for k, v in pixel_counts.items() if k > 0}

    if not tumor_counts:
        return 0  # All background

    # Most common non-zero class
    return max(tumor_counts, key=tumor_counts.get)


def determine_diagnosis(patch_predictions: List[int]) -> int:
    """Majority vote among tumor patches for image-level diagnosis."""
    if not patch_predictions or all(p == 0 for p in patch_predictions):
        return 0  # Healthy

    tumor_preds = [p for p in patch_predictions if p > 0]
    if not tumor_preds:
        return 0

    counts = Counter(tumor_preds)
    return counts.most_common(1)[0][0]


def save_case_predictions(output_dir: Path, case_id: str,
                          patch_info: List[Tuple[int, int, int]],
                          diagnosis: int, patch_size: int,
                          image_shape: Tuple[int, ...]):
    """Save per-case patch predictions to JSON (same schema as classification models)."""
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


def visualize_predictions(viz_dir: Path, case_id: str, image: np.ndarray,
                          patch_info: List[Tuple[int, int, int]], patch_size: int,
                          diagnosis: int):
    """Create 2-panel visualization: original+overlay | prediction grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    h, w = image.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Original image with colored grid overlay
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

    # Right: Prediction grid (solid colors)
    pred_image = np.zeros((h, w, 3), dtype=np.uint8)
    for pred, x, y in patch_info:
        y_end = min(y + patch_size, h)
        x_end = min(x + patch_size, w)
        pred_image[y:y_end, x:x_end] = COLOR_MAP[pred]

    axes[1].imshow(pred_image)

    # Infer ground truth from case_id prefix
    gt = infer_ground_truth(case_id)
    title = f'Predicted: {DIAGNOSIS_NAMES[diagnosis]}'
    if gt is not None:
        title += f' | True: {DIAGNOSIS_NAMES[gt]}'
        title += f' | {"CORRECT" if diagnosis == gt else "INCORRECT"}'
    axes[1].set_title(title, fontsize=12)
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


def infer_ground_truth(case_id: str) -> int:
    """Infer ground truth class from filename prefix."""
    lower = case_id.lower()
    if lower.startswith('cho'):
        return 3
    elif lower.startswith('hcc'):
        return 2
    elif lower.startswith('metastatic'):
        return 1
    return None


def process_dataset(dataset_id: int, dataset_name: str,
                    predictions_dir: Path, images_dir: Path,
                    output_dir: Path, patch_size: int,
                    skip_visualization: bool):
    """Process one dataset: convert nnUNet predictions to patch classification format."""
    ds_key = f"Dataset{dataset_id}_{dataset_name}"
    pred_dir = predictions_dir / ds_key / "ensemble"
    out_dir = output_dir / ds_key / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        logger.warning(f"Prediction directory not found: {pred_dir}")
        return

    # Find all prediction PNGs
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        logger.warning(f"No prediction PNGs in {pred_dir}")
        return

    logger.info(f"Processing {ds_key}: {len(pred_files)} predictions")

    if not skip_visualization:
        viz_dir = out_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    all_diagnoses = {}

    for pred_file in pred_files:
        case_id = pred_file.stem

        # Load nnUNet prediction mask
        mask = np.array(Image.open(pred_file).convert('L'))

        # Load original image for shape and visualization
        img_path = images_dir / f"{case_id}.jpg"
        if not img_path.exists():
            logger.warning(f"Original image not found: {img_path}")
            continue
        original = np.array(Image.open(img_path).convert('RGB'))

        h, w = mask.shape[:2]

        # Extract patch grid and classify each patch
        coords = extract_patch_grid(h, w, patch_size)
        patch_info = []  # (prediction, x, y)

        for x, y in coords:
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch_mask = mask[y:y_end, x:x_end]
            pred_class = classify_patch(patch_mask)
            patch_info.append((pred_class, x, y))

        # Image-level diagnosis
        patch_preds = [p for p, _, _ in patch_info]
        diagnosis = determine_diagnosis(patch_preds)

        # Save per-case JSON
        save_case_predictions(out_dir, case_id, patch_info, diagnosis,
                              patch_size, (h, w, 3))

        all_diagnoses[case_id] = diagnosis

        # Visualization
        if not skip_visualization:
            visualize_predictions(viz_dir, case_id, original, patch_info,
                                  patch_size, diagnosis)

    # Save diagnosis_predictions.json
    diag_data = {
        case_id: {
            'diagnosis': diag,
            'diagnosis_name': DIAGNOSIS_NAMES[diag]
        }
        for case_id, diag in all_diagnoses.items()
    }

    with open(out_dir / "diagnosis_predictions.json", 'w') as f:
        json.dump(diag_data, f, indent=2)

    logger.info(f"  Saved {len(all_diagnoses)} diagnosis predictions to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert nnUNet pixel predictions to patch-based classification format')
    parser.add_argument('--predictions_dir', type=str,
                        default='predictions_external/nnunet',
                        help='Directory with nnUNet prediction PNGs')
    parser.add_argument('--images_dir', type=str,
                        default='data/external_images',
                        help='Directory with original JPG images')
    parser.add_argument('--output_dir', type=str,
                        default='predictions_external/nnunet_patch',
                        help='Output directory for patch classification results')
    parser.add_argument('--patch_size', type=int, default=384,
                        help='Patch size for grid extraction (default: 384)')
    parser.add_argument('--datasets', nargs='+', type=int,
                        default=[100, 101, 102, 103, 104],
                        help='Dataset IDs to process')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip creating visualizations')
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"nnUNet Patch Classification")
    logger.info(f"  Predictions: {predictions_dir}")
    logger.info(f"  Images: {images_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Patch size: {args.patch_size}")
    logger.info(f"  Datasets: {args.datasets}")

    for ds_id in args.datasets:
        if ds_id not in DATASETS:
            logger.warning(f"Unknown dataset ID: {ds_id}")
            continue
        process_dataset(ds_id, DATASETS[ds_id],
                        predictions_dir, images_dir, output_dir,
                        args.patch_size, args.skip_visualization)

    logger.info("Done.")


if __name__ == '__main__':
    main()
