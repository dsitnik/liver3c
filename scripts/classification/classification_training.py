#!/usr/bin/env python3
"""
Classification Training Script with Scientific Rigor
Patch-based liver cancer classification using timm models.
Mirrors nnunet_training.py structure for direct comparison.

Features:
- Reproducibility: Seeds, deterministic mode, version tracking
- Monitoring: Training progress, convergence detection, failure handling
- Documentation: Complete metadata logging for publication

Usage:
    python classification_training.py --dataset_id 100 --dataset_name Liver1 --model convnextv2
    python classification_training.py --dataset_id 100 --dataset_name Liver1 --all
"""

import os
import json
import argparse
import subprocess
import sys
import time
import logging
import platform
import socket
import random
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from functools import lru_cache
import shutil
import io

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import timm
from sklearn.metrics import confusion_matrix

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available, using basic transforms")

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Available timm models for classification
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


def get_train_transforms(mean: List[float], std: List[float]):
    """Get training data augmentations with dataset-specific normalization"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-45, 45),
            p=0.5,
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=1, p=0.3),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.4),
            A.ISONoise(p=0.4),
            A.MultiplicativeNoise(p=0.2),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(mean: List[float], std: List[float]):
    """Get validation data augmentations (only normalization)"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class LiverCancerPatchDataset(Dataset):
    """PyTorch Dataset for patch-based liver cancer classification"""

    # Class-level image cache (shared across instances)
    _image_cache = {}
    _cache_max_size = 200  # Maximum number of images to cache

    def __init__(self,
                 images_dir: Path,
                 labels_dir: Path,
                 case_ids: List[str],
                 patch_size: int = 384,
                 min_cancer_threshold: float = 0.10,
                 transform=None,
                 mode: str = 'train',
                 max_background_ratio: float = 2.0,
                 normalization_stats: Tuple[List[float], List[float]] = None):
        """
        Args:
            images_dir: Path to imagesTr directory
            labels_dir: Path to labelsTr directory
            case_ids: List of case IDs (e.g., ['metastatic_2', 'hcc_1a'])
            patch_size: Size of patches to extract (default: 384)
            min_cancer_threshold: Minimum cancer % to keep patch (default: 0.10)
            transform: Albumentations transform
            mode: 'train' or 'val' (val uses non-overlapping patches)
            max_background_ratio: Max ratio of background to avg cancer class (default: 2.0)
            normalization_stats: Tuple of (mean, std) for normalization fallback
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.min_cancer_threshold = min_cancer_threshold
        self.transform = transform
        self.mode = mode
        self.max_background_ratio = max_background_ratio
        self.normalization_stats = normalization_stats or ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # Precompute valid patches
        logger.info(f"Precomputing valid patches for {len(case_ids)} cases ({mode})...")
        self.all_patches = self._precompute_valid_patches()

        # Balance background class for training
        if mode == 'train':
            self.all_patches = self._balance_background_patches(self.all_patches)

        logger.info(f"Final patch count: {len(self.all_patches)}")

        # Log class distribution
        self._log_class_distribution()

    def _load_image_and_label(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load 3-channel image and label for a case with caching"""
        cache_key = (str(self.images_dir), case_id)

        if cache_key in LiverCancerPatchDataset._image_cache:
            return LiverCancerPatchDataset._image_cache[cache_key]

        # Load 3 channel images
        channels = []
        for ch in range(3):
            img_path = self.images_dir / f"{case_id}_{ch:04d}.png"
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            channel = np.array(Image.open(img_path))
            channels.append(channel)

        # Stack channels: (H, W, 3)
        image = np.stack(channels, axis=-1)

        # Load label
        label_path = self.labels_dir / f"{case_id}.png"
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")
        label = np.array(Image.open(label_path))

        # Cache management: remove oldest if cache is full
        if len(LiverCancerPatchDataset._image_cache) >= LiverCancerPatchDataset._cache_max_size:
            # Remove first (oldest) item
            oldest_key = next(iter(LiverCancerPatchDataset._image_cache))
            del LiverCancerPatchDataset._image_cache[oldest_key]

        LiverCancerPatchDataset._image_cache[cache_key] = (image, label)

        return image, label

    def _validate_single_cancer_type(self, label: np.ndarray, case_id: str):
        """Validate that image contains only one type of cancer (or none)"""
        unique_labels = np.unique(label)
        cancer_labels = [l for l in unique_labels if l > 0]

        if len(cancer_labels) > 1:
            raise ValueError(
                f"Multiple cancer types found in {case_id}: {cancer_labels}. "
                f"Each image should contain only one cancer type."
            )

    def _classify_patch(self, label_patch: np.ndarray) -> Optional[int]:
        """
        Classify a patch based on label content.

        Returns:
            - 0 (background): if 0% cancer (pure background)
            - 1-3 (cancer class): if >=10% cancer (dominant class)
            - None: if >0% but <10% cancer (discard)
        """
        total_pixels = label_patch.size
        cancer_pixels = np.sum(label_patch > 0)
        cancer_percentage = cancer_pixels / total_pixels

        if cancer_percentage == 0:
            # Pure background - KEEP
            return 0
        elif cancer_percentage < self.min_cancer_threshold:
            # Borderline patch - DISCARD
            return None
        else:
            # Cancer patch - return dominant cancer class
            cancer_mask = label_patch > 0
            cancer_values = label_patch[cancer_mask]
            # np.bincount needs non-negative integers, cancer_values are 1, 2, or 3
            counts = np.bincount(cancer_values, minlength=4)
            # Return the class with most pixels (excluding background at index 0)
            return int(np.argmax(counts[1:]) + 1)

    def _precompute_valid_patches(self) -> List[Tuple[str, int, int, int]]:
        """
        Precompute all valid patch locations for all cases.

        For training: 50% overlapping patches for more samples
        For validation: Non-overlapping patches for unbiased evaluation

        Returns:
            List of (case_id, x, y, class_label) tuples
        """
        all_patches = []

        # Use non-overlapping for validation to avoid inflated metrics
        if self.mode == 'val':
            step = self.patch_size  # No overlap
        else:
            step = self.patch_size // 2  # 50% overlap for training

        for case_id in self.case_ids:
            try:
                _, label = self._load_image_and_label(case_id)

                # Validate single cancer type
                self._validate_single_cancer_type(label, case_id)

                h, w = label.shape

                for y in range(0, h - self.patch_size + 1, step):
                    for x in range(0, w - self.patch_size + 1, step):
                        label_patch = label[y:y + self.patch_size, x:x + self.patch_size]
                        patch_class = self._classify_patch(label_patch)

                        if patch_class is not None:
                            all_patches.append((case_id, x, y, patch_class))

            except Exception as e:
                logger.error(f"Error processing case {case_id}: {e}")
                raise

        return all_patches

    def _balance_background_patches(self, patches: List[Tuple[str, int, int, int]]) -> List[Tuple[str, int, int, int]]:
        """
        Balance background patches to avoid class imbalance.

        Limits background patches to max_background_ratio times the average cancer class count.
        """
        # Count patches per class
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        class_patches = {0: [], 1: [], 2: [], 3: []}

        for patch in patches:
            class_label = patch[3]
            class_counts[class_label] += 1
            class_patches[class_label].append(patch)

        # Calculate average cancer class count
        cancer_counts = [class_counts[c] for c in [1, 2, 3] if class_counts[c] > 0]
        if not cancer_counts:
            logger.warning("No cancer patches found!")
            return patches

        avg_cancer_count = np.mean(cancer_counts)
        max_background = int(avg_cancer_count * self.max_background_ratio)

        original_bg_count = class_counts[0]

        # Subsample background if needed
        if class_counts[0] > max_background:
            # Randomly sample background patches
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            indices = rng.choice(len(class_patches[0]), size=max_background, replace=False)
            class_patches[0] = [class_patches[0][i] for i in indices]
            logger.info(f"Background patches reduced: {original_bg_count} -> {max_background} "
                       f"(max_ratio={self.max_background_ratio}, avg_cancer={avg_cancer_count:.0f})")

        # Recombine all patches
        balanced_patches = []
        for cls in range(4):
            balanced_patches.extend(class_patches[cls])

        # Shuffle to mix classes
        rng = np.random.RandomState(42)
        rng.shuffle(balanced_patches)

        return balanced_patches

    def _log_class_distribution(self):
        """Log the class distribution of patches"""
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _, _, _, class_label in self.all_patches:
            class_counts[class_label] += 1

        logger.info(f"Patch class distribution ({self.mode}):")
        for cls, count in class_counts.items():
            pct = count / len(self.all_patches) * 100 if self.all_patches else 0
            logger.info(f"  {CLASS_NAMES[cls]:12s}: {count:6d} ({pct:5.1f}%)")

    def compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling class imbalance"""
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _, _, _, class_label in self.all_patches:
            class_counts[class_label] += 1

        total = sum(class_counts.values())
        # Inverse frequency weighting
        weights = []
        for cls in range(4):
            if class_counts[cls] > 0:
                weights.append(total / (4 * class_counts[cls]))
            else:
                weights.append(1.0)

        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler"""
        class_weights = self.compute_class_weights()
        sample_weights = []
        for _, _, _, class_label in self.all_patches:
            sample_weights.append(class_weights[class_label].item())
        return sample_weights

    def __len__(self) -> int:
        return len(self.all_patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        case_id, x, y, class_label = self.all_patches[idx]

        # Load image
        image, _ = self._load_image_and_label(case_id)

        # Extract patch
        patch = image[y:y + self.patch_size, x:x + self.patch_size, :]

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=patch)
            patch = transformed['image']
        else:
            # Basic transform: normalize and convert to tensor using dataset stats
            mean, std = self.normalization_stats
            patch = patch.astype(np.float32) / 255.0
            patch = (patch - np.array(mean)) / np.array(std)
            patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()

        return patch, class_label


class ClassificationTrainerRigorous:
    """Classification training manager with scientific rigor and reproducibility"""

    def __init__(self, config: Dict, dataset_id: int, dataset_name: str,
                 model_name: str, random_seed: int = 42):
        self.config = config
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_full_name = MODELS.get(model_name, model_name)
        self.dataset_full_name = f"Dataset{dataset_id:03d}_{dataset_name}"
        self.random_seed = random_seed

        # Training session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_start_time = None
        self.fold_results = {}
        self.current_fold = None

        # Normalization stats (loaded later)
        self.normalization_mean = None
        self.normalization_std = None

        # Set up environment
        self.setup_environment()
        self.setup_paths()
        self.log_system_info()
        self.log_software_versions()

        # Load normalization statistics from dataset fingerprint
        self._load_normalization_stats()

    def _load_normalization_stats(self):
        """Load normalization statistics from nnU-Net dataset fingerprint"""
        self.normalization_mean, self.normalization_std = load_normalization_stats(
            self.preprocessed_path
        )

        # Save to metadata
        norm_info = {
            'mean': self.normalization_mean,
            'std': self.normalization_std,
            'source': str(self.preprocessed_path / "dataset_fingerprint.json")
        }
        with open(self.metadata_path / 'normalization_stats.json', 'w') as f:
            json.dump(norm_info, f, indent=2)

    def reset_seeds(self, seed: int = None):
        """Reset all random seeds for reproducibility at fold start"""
        if seed is None:
            seed = self.random_seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Clear image cache for clean fold start
        LiverCancerPatchDataset._image_cache.clear()

        logger.info(f"Reset all random seeds to {seed}")

    def setup_environment(self):
        """Setup environment variables with reproducibility"""
        # Reproducibility: Set environment variables for deterministic behavior
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() and
                                   self.config.get('device', {}).get('use_cuda', True) else 'cpu')

        logger.info("=" * 80)
        logger.info("Classification Training Environment Setup:")
        logger.info(f"  Random seed:         {self.random_seed}")
        logger.info(f"  Device:              {self.device}")
        logger.info(f"  Model:               {self.model_name} ({self.model_full_name})")
        logger.info(f"  Session ID:          {self.session_id}")
        logger.info("=" * 80)

    def setup_paths(self):
        """Setup dataset and output paths"""
        current_dir = Path.cwd()

        # nnUNet paths (for reading data)
        self.nnunet_raw = current_dir / "nnUNet_raw"
        self.nnunet_preprocessed = current_dir / "nnUNet_preprocessed"

        # Dataset paths
        self.dataset_path = self.nnunet_raw / self.dataset_full_name
        self.images_dir = self.dataset_path / "imagesTr"
        self.labels_dir = self.dataset_path / "labelsTr"
        self.preprocessed_path = self.nnunet_preprocessed / self.dataset_full_name

        # Output paths
        self.results_base = current_dir / "classification_results"
        self.results_path = self.results_base / self.dataset_full_name / self.model_name
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Metadata path
        self.metadata_path = current_dir / "classification_metadata" / self.dataset_full_name / self.session_id
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        logger.info("Paths:")
        logger.info(f"  Dataset:     {self.dataset_path}")
        logger.info(f"  Images:      {self.images_dir}")
        logger.info(f"  Labels:      {self.labels_dir}")
        logger.info(f"  Results:     {self.results_path}")
        logger.info(f"  Metadata:    {self.metadata_path}")

    def log_system_info(self):
        """Log system information for reproducibility"""
        system_info = {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'random_seed': self.random_seed,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'model_full_name': self.model_full_name
        }

        # GPU info
        if torch.cuda.is_available():
            system_info['cuda_available'] = True
            system_info['cuda_version'] = torch.version.cuda
            system_info['gpu_count'] = torch.cuda.device_count()
            system_info['gpu_names'] = [torch.cuda.get_device_name(i)
                                        for i in range(torch.cuda.device_count())]
        else:
            system_info['cuda_available'] = False

        # Save system info
        with open(self.metadata_path / 'system_info.json', 'w') as f:
            json.dump(system_info, f, indent=2)

        logger.info("\nSystem Information:")
        logger.info(f"  Hostname:     {system_info['hostname']}")
        logger.info(f"  Platform:     {system_info['platform']}")
        logger.info(f"  Python:       {system_info['python_version']}")
        logger.info(f"  CPUs:         {system_info['cpu_count']}")
        if system_info.get('cuda_available'):
            logger.info(f"  CUDA:         {system_info['cuda_version']}")
            logger.info(f"  GPUs:         {system_info['gpu_count']}")
            for i, gpu in enumerate(system_info.get('gpu_names', [])):
                logger.info(f"    GPU {i}:     {gpu}")

    def log_software_versions(self):
        """Log software versions for reproducibility"""
        versions = {
            'torch': torch.__version__,
            'timm': timm.__version__,
        }

        # Check other packages
        for package in ['numpy', 'albumentations', 'PIL', 'sklearn']:
            try:
                if package == 'PIL':
                    from PIL import __version__ as pil_version
                    versions['pillow'] = pil_version
                elif package == 'sklearn':
                    import sklearn
                    versions['scikit-learn'] = sklearn.__version__
                else:
                    module = __import__(package)
                    versions[package] = getattr(module, '__version__', 'Unknown')
            except ImportError:
                versions[package] = 'Not installed'

        # Save versions
        with open(self.metadata_path / 'software_versions.json', 'w') as f:
            json.dump(versions, f, indent=2)

        logger.info("\nSoftware Versions:")
        for pkg, ver in versions.items():
            logger.info(f"  {pkg:15s} {ver}")

    def validate_dataset(self) -> Dict:
        """Validate that the dataset exists and is properly formatted"""
        logger.info("\nValidating dataset...")

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found: {self.dataset_path}")

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")

        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")

        dataset_json = self.dataset_path / "dataset.json"
        if not dataset_json.exists():
            raise ValueError(f"dataset.json not found: {dataset_json}")

        with open(dataset_json) as f:
            dataset_info = json.load(f)

        logger.info("Dataset validated:")
        logger.info(f"  Training cases: {dataset_info.get('numTraining', 'Unknown')}")
        logger.info(f"  Test cases:     {dataset_info.get('numTest', 'Unknown')}")
        logger.info(f"  Labels:         {list(dataset_info.get('labels', {}).keys())}")

        # Save dataset info to metadata
        with open(self.metadata_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        return dataset_info

    def load_splits(self) -> List[Dict]:
        """Load CV splits from splits_final.json"""
        splits_file = self.preprocessed_path / "splits_final.json"

        if not splits_file.exists():
            # Try raw directory
            splits_file = self.dataset_path / "splits_final.json"

        if not splits_file.exists():
            raise FileNotFoundError(f"splits_final.json not found in {self.preprocessed_path} or {self.dataset_path}")

        with open(splits_file) as f:
            splits = json.load(f)

        logger.info(f"Loaded {len(splits)} folds from {splits_file}")
        return splits

    def get_fold_numbers(self) -> List[int]:
        """Get fold numbers from splits file"""
        splits = self.load_splits()
        fold_numbers = list(range(len(splits)))
        logger.info(f"Available folds: {fold_numbers}")
        return fold_numbers

    def check_checkpoint_exists(self, fold: int) -> bool:
        """Check if a checkpoint exists for this fold"""
        fold_path = self.results_path / f"fold_{fold}"
        checkpoint_best = fold_path / "checkpoint_best.pth"
        checkpoint_final = fold_path / "checkpoint_final.pth"
        return checkpoint_best.exists() or checkpoint_final.exists()

    def check_fold_completed(self, fold: int) -> bool:
        """Check if fold training is fully completed (early stopped or max epochs reached)"""
        fold_path = self.results_path / f"fold_{fold}"
        # confusion_matrix_final.json is only saved when training completes
        return (fold_path / "confusion_matrix_final.json").exists()

    def create_model(self, pretrained: bool = False) -> nn.Module:
        """Create timm model for classification"""
        logger.info(f"Creating model: {self.model_full_name}")
        logger.info(f"  Pretrained weights: {'ImageNet' if pretrained else 'None (training from scratch)'}")

        model = timm.create_model(
            self.model_full_name,
            pretrained=pretrained,
            num_classes=4,  # background, metastatic, hcc, cho
            in_chans=3,
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total parameters:     {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        return model.to(self.device)

    def save_checkpoint(self, model: nn.Module, optimizer, scheduler, epoch: int,
                       best_val_dice: float, filename: str):
        """Save checkpoint with metadata"""
        fold_path = self.results_path / f"fold_{self.current_fold}"
        fold_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_full_name': self.model_full_name,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': self.config,
            'class_names': CLASS_NAMES,
            'patch_size': self.config.get('patch_size', 384),
            'random_seed': self.random_seed,
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset_name,
            'fold': self.current_fold,
            'best_val_dice': best_val_dice,
        }

        save_path = fold_path / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, model: nn.Module, optimizer, scheduler, fold: int) -> Tuple[int, float]:
        """Load checkpoint and return (start_epoch, best_val_dice)"""
        fold_path = self.results_path / f"fold_{fold}"

        # Try checkpoint_best first, then checkpoint_final
        for checkpoint_name in ['checkpoint_best.pth', 'checkpoint_final.pth']:
            checkpoint_path = fold_path / checkpoint_name
            if checkpoint_path.exists():
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and checkpoint.get('scheduler_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                start_epoch = checkpoint['epoch'] + 1
                # Support both old (best_val_f1) and new (best_val_dice) checkpoint formats
                best_val_dice = checkpoint.get('best_val_dice', checkpoint.get('best_val_f1', 0.0))

                logger.info(f"Resumed from epoch {checkpoint['epoch']}, best Dice: {best_val_dice:.4f}")
                return start_epoch, best_val_dice

        return 0, 0.0

    def compute_dice_score(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute macro-averaged Dice score (equivalent to F1 for classification)"""
        # Dice = 2*TP / (2*TP + FP + FN) = F1 score
        dice_per_class = []
        for cls in range(4):
            pred_cls = (preds == cls)
            label_cls = (labels == cls)

            tp = np.sum(pred_cls & label_cls)
            fp = np.sum(pred_cls & ~label_cls)
            fn = np.sum(~pred_cls & label_cls)

            if (2 * tp + fp + fn) > 0:
                dice = (2 * tp) / (2 * tp + fp + fn)
            else:
                dice = 1.0  # Both empty = perfect match
            dice_per_class.append(dice)

        return np.mean(dice_per_class)

    def train_epoch(self, model: nn.Module, dataloader: DataLoader,
                   criterion: nn.Module, optimizer,
                   grad_clip_norm: float = 1.0) -> Tuple[float, float]:
        """Train for one epoch with gradient clipping"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                # Compute running Dice
                running_dice = self.compute_dice_score(
                    np.array(all_preds), np.array(all_labels)
                )
                logger.info(f"    Batch {batch_idx + 1}/{len(dataloader)}, "
                           f"Loss: {loss.item():.4f}, "
                           f"Dice: {running_dice:.4f}")

        avg_loss = total_loss / len(dataloader)
        dice = self.compute_dice_score(np.array(all_preds), np.array(all_labels))
        return avg_loss, dice

    def validate_epoch(self, model: nn.Module, dataloader: DataLoader,
                      criterion: nn.Module) -> Tuple[float, float, np.ndarray, Dict]:
        """Validate with comprehensive metrics including per-class Dice scores"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

        # Compute per-class Dice scores (Dice = F1 for binary classification per class)
        per_class_dice = {}
        dice_scores = []
        for cls in range(4):
            pred_cls = (all_preds == cls)
            label_cls = (all_labels == cls)

            tp = np.sum(pred_cls & label_cls)
            fp = np.sum(pred_cls & ~label_cls)
            fn = np.sum(~pred_cls & label_cls)

            if (2 * tp + fp + fn) > 0:
                dice = (2 * tp) / (2 * tp + fp + fn)
            else:
                dice = 1.0  # Both empty = perfect match

            per_class_dice[CLASS_NAMES[cls]] = float(dice)
            dice_scores.append(dice)

        # Macro-averaged Dice score
        val_dice = np.mean(dice_scores)

        return avg_loss, val_dice, conf_matrix, per_class_dice

    def _save_confusion_matrix(self, conf_matrix: np.ndarray, fold_path: Path,
                               epoch: int, final: bool = False):
        """Save confusion matrix to file"""
        if final:
            filename = 'confusion_matrix_final.json'
        else:
            filename = 'confusion_matrix_best.json'

        cm_data = {
            'epoch': epoch + 1,
            'class_names': list(CLASS_NAMES.values()),
            'matrix': conf_matrix.tolist(),
            'per_class_metrics': {}
        }

        # Calculate per-class precision, recall, F1
        for i, class_name in CLASS_NAMES.items():
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            cm_data['per_class_metrics'][class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(conf_matrix[i, :].sum())
            }

        with open(fold_path / filename, 'w') as f:
            json.dump(cm_data, f, indent=2)

        logger.info(f"  Saved confusion matrix: {fold_path / filename}")

    def train_single_fold(self, fold: int, resume: bool = None):
        """Train a single fold with comprehensive logging"""
        self.current_fold = fold

        # Reset seeds for reproducibility at fold start
        fold_seed = self.random_seed + fold  # Different but reproducible seed per fold
        self.reset_seeds(fold_seed)

        # Auto-detect resume
        if resume is None:
            resume = self.check_checkpoint_exists(fold)
            if resume:
                logger.info(f"Found existing checkpoint for fold {fold}, will resume")

        logger.info("\n" + "=" * 80)
        logger.info(f"{'RESUMING' if resume else 'STARTING'} TRAINING")
        logger.info(f"Model:         {self.model_name} ({self.model_full_name})")
        logger.info(f"Fold:          {fold}")
        logger.info(f"Device:        {self.device}")
        logger.info(f"Fold seed:     {fold_seed}")
        logger.info("=" * 80)

        fold_start_time = time.time()

        # Load splits
        splits = self.load_splits()
        train_cases = splits[fold]['train']
        val_cases = splits[fold]['val']

        logger.info(f"Train cases: {len(train_cases)}, Val cases: {len(val_cases)}")

        # Get config values
        patch_size = self.config.get('patch_size', 384)
        batch_size = self.config.get('batch_size', 32)
        max_epochs = self.config.get('max_epochs', 100)
        learning_rate = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        num_workers = self.config.get('num_workers', 4)
        min_cancer_threshold = self.config.get('min_cancer_threshold', 0.10)
        max_background_ratio = self.config.get('max_background_ratio', 2.0)
        grad_clip_norm = self.config.get('grad_clip_norm', 1.0)

        # Create datasets with dataset-specific normalization
        train_dataset = LiverCancerPatchDataset(
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            case_ids=train_cases,
            patch_size=patch_size,
            min_cancer_threshold=min_cancer_threshold,
            transform=get_train_transforms(self.normalization_mean, self.normalization_std),
            mode='train',
            max_background_ratio=max_background_ratio,
            normalization_stats=(self.normalization_mean, self.normalization_std)
        )

        val_dataset = LiverCancerPatchDataset(
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            case_ids=val_cases,
            patch_size=patch_size,
            min_cancer_threshold=min_cancer_threshold,
            transform=get_val_transforms(self.normalization_mean, self.normalization_std),
            mode='val',
            max_background_ratio=max_background_ratio,  # Not used in val, but for consistency
            normalization_stats=(self.normalization_mean, self.normalization_std)
        )

        # Create balanced sampler for training
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Create model, optimizer, scheduler
        pretrained = self.config.get('pretrained', False)
        model = self.create_model(pretrained=pretrained)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-7
        )

        # Loss with class weights
        class_weights = train_dataset.compute_class_weights().to(self.device)
        logger.info(f"Class weights: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Resume if needed
        start_epoch = 0
        best_val_dice = 0.0
        if resume:
            start_epoch, best_val_dice = self.load_checkpoint(model, optimizer, scheduler, fold)

        # TensorBoard writer
        fold_path = self.results_path / f"fold_{fold}"
        fold_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(fold_path / 'tensorboard'))

        # Training log file
        log_file = fold_path / 'training_log.csv'
        if not resume or not log_file.exists():
            with open(log_file, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['epoch', 'train_loss', 'train_dice', 'val_loss',
                                   'val_dice', 'lr', 'time_elapsed'])

        # Training loop
        early_stopping_patience = self.config.get('early_stopping_patience', 20)
        epochs_without_improvement = 0

        try:
            for epoch in range(start_epoch, max_epochs):
                epoch_start = time.time()

                logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")
                logger.info("-" * 40)

                # Train with gradient clipping
                train_loss, train_dice = self.train_epoch(
                    model, train_loader, criterion, optimizer, grad_clip_norm
                )

                # Validate
                val_loss, val_dice, conf_matrix, per_class_dice = self.validate_epoch(
                    model, val_loader, criterion
                )

                # Update scheduler (ReduceLROnPlateau uses val_dice)
                scheduler.step(val_dice)
                current_lr = optimizer.param_groups[0]['lr']

                epoch_time = time.time() - epoch_start

                # Log metrics
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
                logger.info(f"  Val Loss:   {val_loss:.4f}, Val Dice:   {val_dice:.4f}")
                logger.info(f"  Per-class Dice: {per_class_dice}")
                logger.info(f"  LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")

                # TensorBoard logging
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Dice/train', train_dice, epoch)
                writer.add_scalar('Dice/val', val_dice, epoch)
                writer.add_scalar('LR', current_lr, epoch)
                # Per-class Dice to TensorBoard
                for class_name, dice_val in per_class_dice.items():
                    writer.add_scalar(f'Dice_class/{class_name}', dice_val, epoch)

                # CSV logging
                with open(log_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([epoch + 1, train_loss, train_dice, val_loss,
                                       val_dice, current_lr, epoch_time])

                # Checkpointing (use val_dice for best model selection)
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    self.save_checkpoint(model, optimizer, scheduler, epoch, best_val_dice,
                                       'checkpoint_best.pth')
                    epochs_without_improvement = 0
                    logger.info(f"  New best Dice: {best_val_dice:.4f}")

                    # Save confusion matrix for best epoch
                    self._save_confusion_matrix(conf_matrix, fold_path, epoch)
                else:
                    epochs_without_improvement += 1

                # Save latest checkpoint
                self.save_checkpoint(model, optimizer, scheduler, epoch, best_val_dice,
                                   'checkpoint_final.pth')

                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping after {early_stopping_patience} epochs without improvement")
                    break

            fold_elapsed = time.time() - fold_start_time
            logger.info(f"\nFold {fold} completed in {fold_elapsed / 3600:.2f} hours")
            logger.info(f"Best validation Dice: {best_val_dice:.4f}")

            # Save final confusion matrix
            self._save_confusion_matrix(conf_matrix, fold_path, epoch, final=True)

            # Record fold result
            self.fold_results[f"fold_{fold}"] = {
                'status': 'completed',
                'elapsed_time': fold_elapsed,
                'best_val_dice': best_val_dice,
                'final_epoch': epoch + 1
            }

        except Exception as e:
            fold_elapsed = time.time() - fold_start_time
            logger.error(f"Training failed for fold {fold}: {e}")
            self.fold_results[f"fold_{fold}"] = {
                'status': 'failed',
                'elapsed_time': fold_elapsed,
                'error': str(e)
            }
            raise

        finally:
            writer.close()

    def train_all_folds(self, single_fold: int = None, fold_list: List[int] = None,
                       resume: bool = None):
        """Train all folds, a single fold, or a list of folds"""
        self.training_start_time = time.time()

        # Determine folds to train
        if single_fold is not None:
            fold_numbers = [single_fold]
            logger.info(f"Training only fold {single_fold}")
        elif fold_list is not None:
            fold_numbers = fold_list
            logger.info(f"Training folds: {fold_list}")
        else:
            fold_numbers = self.get_fold_numbers()
            logger.info(f"Training all folds: {fold_numbers}")

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SESSION STARTED")
        logger.info(f"Model:              {self.model_name}")
        logger.info(f"Session ID:         {self.session_id}")
        logger.info(f"Folds to train:     {fold_numbers}")
        logger.info("=" * 80 + "\n")

        successful_folds = []
        failed_folds = []
        skipped_folds = []

        for fold in fold_numbers:
            # Skip folds that are already completed (early stopped or max epochs)
            if self.check_fold_completed(fold):
                logger.info(f"Fold {fold} already completed, skipping")
                skipped_folds.append(fold)
                continue

            try:
                self.train_single_fold(fold, resume=resume)
                successful_folds.append(fold)
            except Exception as e:
                failed_folds.append(fold)
                logger.error(f"Failed to train fold {fold}: {e}")

                if not self.config.get('continue_on_error', False):
                    logger.error("Stopping due to error (continue_on_error=False)")
                    raise
                else:
                    logger.warning("Continuing with next fold (continue_on_error=True)")

        # Create summary
        self.create_training_summary(successful_folds, failed_folds, skipped_folds)

        total_elapsed = time.time() - self.training_start_time
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SESSION COMPLETED")
        logger.info(f"Total time:         {total_elapsed / 3600:.2f} hours")
        if skipped_folds:
            logger.info(f"Skipped folds:      {skipped_folds} (already completed)")
        logger.info(f"Successful folds:   {successful_folds}")
        if failed_folds:
            logger.info(f"Failed folds:       {failed_folds}")
        logger.info(f"Results:            {self.results_path}")
        logger.info("=" * 80 + "\n")

        return successful_folds

    def create_training_summary(self, successful_folds: List[int], failed_folds: List[int],
                                skipped_folds: List[int] = None):
        """Create comprehensive training summary report"""
        if skipped_folds is None:
            skipped_folds = []
        total_elapsed = time.time() - self.training_start_time if self.training_start_time else 0

        summary = {
            'dataset': {
                'id': self.dataset_id,
                'name': self.dataset_name,
                'full_name': self.dataset_full_name
            },
            'model': {
                'name': self.model_name,
                'full_name': self.model_full_name
            },
            'training_session': {
                'session_id': self.session_id,
                'start_time': self.training_start_time,
                'end_time': time.time(),
                'total_elapsed_hours': total_elapsed / 3600,
                'random_seed': self.random_seed
            },
            'folds': {
                'successful': successful_folds,
                'failed': failed_folds,
                'skipped': skipped_folds,
                'total': len(successful_folds) + len(failed_folds) + len(skipped_folds)
            },
            'fold_results': self.fold_results,
            'config': self.config,
            'paths': {
                'results': str(self.results_path),
                'metadata': str(self.metadata_path)
            }
        }

        summary_file = self.metadata_path / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Human-readable summary
        readable_summary = self.metadata_path / 'training_summary.txt'
        with open(readable_summary, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLASSIFICATION TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Dataset:           {self.dataset_full_name}\n")
            f.write(f"Model:             {self.model_name} ({self.model_full_name})\n")
            f.write(f"Session ID:        {self.session_id}\n")
            f.write(f"Total time:        {total_elapsed / 3600:.2f} hours\n")
            f.write(f"Random seed:       {self.random_seed}\n")
            f.write(f"\nResults directory: {self.results_path}\n")
            f.write(f"Metadata directory: {self.metadata_path}\n\n")

            f.write("Fold Results:\n")
            f.write("-" * 80 + "\n")
            if skipped_folds:
                f.write(f"  Skipped (already completed): {skipped_folds}\n")
            for fold_name, result in self.fold_results.items():
                status = result['status']
                elapsed = result.get('elapsed_time', 0) / 3600
                dice = result.get('best_val_dice', 0)
                f.write(f"  {fold_name:15s} {status:10s} Dice={dice:.4f} {elapsed:.2f} hours\n")

        logger.info(f"Training summary saved: {summary_file}")


def train_all_models(config: Dict, dataset_id: int, dataset_name: str,
                    single_fold: int = None, fold_list: List[int] = None,
                    resume: bool = None, random_seed: int = 42):
    """Train all 5 models sequentially"""
    logger.info("=" * 80)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 80)

    all_results = {}

    for model_name in MODELS.keys():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Starting model: {model_name}")
        logger.info("=" * 80)

        try:
            trainer = ClassificationTrainerRigorous(
                config, dataset_id, dataset_name, model_name, random_seed
            )
            trainer.validate_dataset()
            successful_folds = trainer.train_all_folds(
                single_fold=single_fold,
                fold_list=fold_list,
                resume=resume
            )
            all_results[model_name] = {
                'status': 'completed',
                'successful_folds': successful_folds
            }
        except Exception as e:
            logger.error(f"Failed to train model {model_name}: {e}")
            all_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
            if not config.get('continue_on_error', False):
                raise

    logger.info("\n" + "=" * 80)
    logger.info("ALL MODELS TRAINING SUMMARY")
    logger.info("=" * 80)
    for model_name, result in all_results.items():
        status = result['status']
        if status == 'completed':
            logger.info(f"  {model_name:20s}: {status} (folds: {result['successful_folds']})")
        else:
            logger.info(f"  {model_name:20s}: {status} ({result.get('error', 'Unknown error')})")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Classification Training Script with Scientific Rigor and Reproducibility'
    )

    # Dataset arguments (mirror nnunet_training.py)
    parser.add_argument('--dataset_id', type=int, required=True,
                       help='Dataset ID (3-digit number, e.g., 100-104)')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name (auto-detected from DATASETS if not provided)')
    parser.add_argument('--config', type=str, default='classification_config.json',
                       help='Path to configuration file')

    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str, choices=list(MODELS.keys()),
                            help='Single model to train')
    model_group.add_argument('--all', action='store_true',
                            help='Train all 5 models sequentially')

    # Fold arguments (mirror nnunet_training.py)
    parser.add_argument('--fold', type=int,
                       help='Train only a specific fold')
    parser.add_argument('--fold_range', type=str,
                       help='Train a range of folds (e.g., 0-4)')

    # Training control (mirror nnunet_training.py)
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted training')
    parser.add_argument('--force_restart', action='store_true',
                       help='Force restart even if checkpoints exist')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Classification-specific arguments
    parser.add_argument('--patch_size', type=int, default=384,
                       help='Patch size for extraction (default: 384)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--min_cancer_threshold', type=float, default=0.10,
                       help='Minimum cancer percentage to keep patch (default: 0.10)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use ImageNet pretrained weights (default: train from scratch)')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        logger.warning(f"Configuration file not found: {args.config}, using defaults")

    # Override config with command-line arguments
    config['patch_size'] = args.patch_size
    config['batch_size'] = args.batch_size
    config['max_epochs'] = args.epochs
    config['learning_rate'] = args.learning_rate
    config['min_cancer_threshold'] = args.min_cancer_threshold
    config['pretrained'] = args.pretrained

    # Set defaults if not in config
    config.setdefault('weight_decay', 0.01)
    config.setdefault('num_workers', 4)
    config.setdefault('early_stopping_patience', 20)
    config.setdefault('continue_on_error', False)
    config.setdefault('device', {'use_cuda': True})

    # Determine resume behavior
    resume = None
    if args.resume:
        resume = True
        logger.info("Resume mode enabled")
    elif args.force_restart:
        resume = False
        logger.info("Force restart enabled")

    # Auto-detect dataset name from DATASETS mapping if not provided
    dataset_name = args.dataset_name
    if dataset_name is None:
        if args.dataset_id in DATASETS:
            dataset_name = DATASETS[args.dataset_id]
            logger.info(f"Auto-detected dataset name: {dataset_name}")
        else:
            logger.error(f"Dataset ID {args.dataset_id} not in DATASETS mapping. "
                        f"Available: {list(DATASETS.keys())}. "
                        f"Please provide --dataset_name explicitly.")
            sys.exit(1)

    # Parse fold range
    fold_list = None
    if args.fold_range:
        if args.fold:
            logger.error("Cannot specify both --fold and --fold_range")
            sys.exit(1)
        try:
            start, end = map(int, args.fold_range.split('-'))
            fold_list = list(range(start, end + 1))
            logger.info(f"Training folds: {fold_list}")
        except ValueError:
            logger.error(f"Invalid fold range format: {args.fold_range}. Use format like '0-4'")
            sys.exit(1)

    try:
        if args.all:
            # Train all models
            train_all_models(
                config, args.dataset_id, dataset_name,
                single_fold=args.fold,
                fold_list=fold_list,
                resume=resume,
                random_seed=args.random_seed
            )
        else:
            # Train single model
            trainer = ClassificationTrainerRigorous(
                config, args.dataset_id, dataset_name,
                args.model, args.random_seed
            )
            trainer.validate_dataset()
            trainer.train_all_folds(
                single_fold=args.fold,
                fold_list=fold_list,
                resume=resume
            )

        logger.info("\nTraining completed successfully!")

    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
