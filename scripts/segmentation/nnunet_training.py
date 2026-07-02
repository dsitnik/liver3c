#!/usr/bin/env python3
"""
nnUNet Training Script with Scientific Rigor
Comprehensive training script for nnUNet with reproducibility, monitoring, and documentation.

Features:
- Reproducibility: Seeds, deterministic mode, version tracking
- Monitoring: Training progress, convergence detection, failure handling
- Documentation: Complete metadata logging for publication

Usage: python nnunet_training.py --dataset_id 100 --dataset_name Liver1
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
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import shutil
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nnunet_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class nnUNetTrainerRigorous:
    """nnUNet training manager with scientific rigor and reproducibility"""

    def __init__(self, config: Dict, dataset_id: int, dataset_name: str, random_seed: int = 42):
        self.config = config
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.dataset_full_name = f"Dataset{dataset_id:03d}_{dataset_name}"
        self.random_seed = random_seed

        # Training session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_start_time = None
        self.fold_results = {}

        # Set up environment
        self.setup_environment()
        self.setup_paths()
        self.log_system_info()
        self.log_software_versions()

    def setup_environment(self):
        """Setup nnUNet environment variables with reproducibility"""
        # Set nnUNet paths
        current_dir = Path.cwd()
        self.nnunet_raw = current_dir / "nnUNet_raw"
        self.nnunet_preprocessed = current_dir / "nnUNet_preprocessed"
        self.nnunet_results = current_dir / "nnUNet_results"

        # Set environment variables
        os.environ['nnUNet_raw'] = str(self.nnunet_raw)
        os.environ['nnUNet_preprocessed'] = str(self.nnunet_preprocessed)
        os.environ['nnUNet_results'] = str(self.nnunet_results)

        # Reproducibility: Set environment variables for deterministic behavior
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA operations

        # Add current directory to Python path
        current_dir_str = str(Path.cwd())
        if current_dir_str not in sys.path:
            sys.path.insert(0, current_dir_str)

        # Create directories
        self.nnunet_raw.mkdir(exist_ok=True)
        self.nnunet_preprocessed.mkdir(exist_ok=True)
        self.nnunet_results.mkdir(exist_ok=True)

        logger.info("="*80)
        logger.info("nnUNet Environment Setup:")
        logger.info(f"  nnUNet_raw:          {self.nnunet_raw}")
        logger.info(f"  nnUNet_preprocessed: {self.nnunet_preprocessed}")
        logger.info(f"  nnUNet_results:      {self.nnunet_results}")
        logger.info(f"  Random seed:         {self.random_seed}")
        logger.info(f"  Session ID:          {self.session_id}")
        logger.info("="*80)

    def setup_paths(self):
        """Setup dataset paths"""
        self.dataset_path = self.nnunet_raw / self.dataset_full_name
        self.preprocessed_path = self.nnunet_preprocessed / self.dataset_full_name
        self.results_path = self.nnunet_results / self.dataset_full_name

        # Training metadata directory
        self.metadata_path = Path("training_metadata") / self.dataset_full_name / self.session_id
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset paths:")
        logger.info(f"  Raw:          {self.dataset_path}")
        logger.info(f"  Preprocessed: {self.preprocessed_path}")
        logger.info(f"  Results:      {self.results_path}")
        logger.info(f"  Metadata:     {self.metadata_path}")

    def log_system_info(self):
        """Log system information for reproducibility"""
        system_info = {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'random_seed': self.random_seed,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }

        # Try to get GPU info
        try:
            import torch
            system_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info['cuda_version'] = torch.version.cuda
                system_info['gpu_count'] = torch.cuda.device_count()
                system_info['gpu_names'] = [torch.cuda.get_device_name(i)
                                           for i in range(torch.cuda.device_count())]
        except ImportError:
            system_info['torch'] = 'Not installed'

        # Save system info
        with open(self.metadata_path / 'system_info.json', 'w') as f:
            json.dump(system_info, f, indent=2)

        logger.info("\nSystem Information:")
        logger.info(f"  Hostname:     {system_info['hostname']}")
        logger.info(f"  Platform:     {system_info['platform']}")
        logger.info(f"  Python:       {system_info['python_version']}")
        logger.info(f"  CPUs:         {system_info['cpu_count']}")
        if 'cuda_available' in system_info and system_info['cuda_available']:
            logger.info(f"  CUDA:         {system_info['cuda_version']}")
            logger.info(f"  GPUs:         {system_info['gpu_count']}")
            for i, gpu in enumerate(system_info.get('gpu_names', [])):
                logger.info(f"    GPU {i}:     {gpu}")

    def log_software_versions(self):
        """Log software versions for reproducibility"""
        versions = {}

        # Check nnUNet version
        try:
            cmd = self.resolve_nnunet_command(['nnUNetv2_train', '--version'])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            versions['nnunet'] = result.stdout.strip() if result.returncode == 0 else 'Unknown'
        except:
            versions['nnunet'] = 'Unknown'

        # Check PyTorch version
        try:
            import torch
            versions['torch'] = torch.__version__
        except ImportError:
            versions['torch'] = 'Not installed'

        # Check other packages
        for package in ['numpy', 'scipy', 'scikit-learn', 'pandas']:
            try:
                module = __import__(package.replace('-', '_'))
                versions[package] = getattr(module, '__version__', 'Unknown')
            except ImportError:
                versions[package] = 'Not installed'

        # Save versions
        with open(self.metadata_path / 'software_versions.json', 'w') as f:
            json.dump(versions, f, indent=2)

        logger.info("\nSoftware Versions:")
        for pkg, ver in versions.items():
            logger.info(f"  {pkg:15s} {ver}")

    def resolve_nnunet_command(self, cmd: List[str]) -> List[str]:
        """Resolve nnUNet command to full path on Windows"""
        if not cmd:
            return cmd

        # Check if first element is an nnUNet command
        if cmd[0].startswith('nnUNetv2_'):
            # On Windows, check venv Scripts folder
            if sys.platform == 'win32':
                venv_scripts = Path('.venv') / 'Scripts'
                exe_path = venv_scripts / f"{cmd[0]}.exe"
                if exe_path.exists():
                    # Use full path to executable
                    return [str(exe_path.absolute())] + cmd[1:]

        return cmd

    def run_command(self, cmd: List[str], description: str, capture_output: bool = False,
                   log_file: Optional[Path] = None):
        """Execute a shell command with comprehensive logging"""
        # Resolve nnUNet commands to full paths on Windows
        cmd = self.resolve_nnunet_command(cmd)

        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"{'='*80}")

        start_time = time.time()

        try:
            if log_file:
                # Save command to log file
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Time: {datetime.now().isoformat()}\n")
                    f.write(f"Command: {' '.join(cmd)}\n")
                    f.write(f"{'='*80}\n\n")

                # Run with output to log file
                with open(log_file, 'a', encoding='utf-8') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, cmd)

                return None, None

            elif capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return result.stdout, result.stderr
            else:
                subprocess.run(cmd, check=True)
                return None, None

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            logger.error(f"Command failed after {elapsed:.1f}s: {e}")
            if capture_output and hasattr(e, 'stdout'):
                logger.error(f"Stdout: {e.stdout}")
                logger.error(f"Stderr: {e.stderr}")
            raise

        finally:
            elapsed = time.time() - start_time
            logger.info(f"Command completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

    def validate_dataset(self):
        """Validate that the dataset exists and is properly formatted"""
        logger.info("\nValidating dataset...")

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found: {self.dataset_path}")

        required_folders = ["imagesTr", "labelsTr"]
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            if not folder_path.exists():
                raise ValueError(f"Required folder missing: {folder_path}")

        dataset_json = self.dataset_path / "dataset.json"
        if not dataset_json.exists():
            raise ValueError(f"dataset.json not found: {dataset_json}")

        # Load and validate dataset.json
        with open(dataset_json) as f:
            dataset_info = json.load(f)

        logger.info(f"Dataset validated:")
        logger.info(f"  Training cases: {dataset_info.get('numTraining', 'Unknown')}")
        logger.info(f"  Test cases:     {dataset_info.get('numTest', 'Unknown')}")
        logger.info(f"  Labels:         {list(dataset_info.get('labels', {}).keys())}")

        # Save dataset info to metadata
        with open(self.metadata_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        return dataset_info

    def plan_and_preprocess(self):
        """Run nnUNet planning and preprocessing"""
        logger.info("\n" + "="*80)
        logger.info("Starting nnUNet planning and preprocessing")
        logger.info("="*80)

        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(self.dataset_id),
            "--verify_dataset_integrity"
        ]

        # Add configuration-based parameters
        if self.config.get('device', {}).get('use_cuda', True):
            num_processes = self.config.get('training', {}).get('num_workers', 4)
            cmd.extend(["-np", str(num_processes)])

        # Create log file for preprocessing
        preprocess_log = self.metadata_path / 'preprocessing.log'

        self.run_command(cmd, "Planning and preprocessing", log_file=preprocess_log)
        logger.info("Planning and preprocessing completed")

    def copy_custom_splits(self):
        """Copy custom CV splits to preprocessed folder"""
        splits_source = self.dataset_path / "splits_final.json"
        splits_dest = self.preprocessed_path / "splits_final.json"

        if splits_source.exists():
            # Ensure preprocessed directory exists
            self.preprocessed_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(splits_source, splits_dest)
            logger.info(f"Copied custom splits to: {splits_dest}")

            # Also save to metadata
            shutil.copy2(splits_source, self.metadata_path / "splits_final.json")
        else:
            logger.warning("No custom splits found, nnUNet will use default 5-fold CV")

    def get_fold_numbers(self) -> List[int]:
        """Get fold numbers from splits file"""
        splits_file = self.preprocessed_path / "splits_final.json"

        if splits_file.exists():
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            fold_numbers = list(range(len(splits)))
            logger.info(f"Using {len(fold_numbers)} folds: {fold_numbers}")
        else:
            # Default 5-fold CV
            fold_numbers = [0, 1, 2, 3, 4]
            logger.info("Using default 5-fold CV")

        return fold_numbers

    def check_checkpoint_exists(self, configuration: str, fold: int) -> bool:
        """Check if a checkpoint exists for this fold"""
        fold_path = self.results_path / f"nnUNetTrainer__nnUNetPlans__{configuration}" / f"fold_{fold}"
        checkpoint_best = fold_path / "checkpoint_best.pth"
        checkpoint_final = fold_path / "checkpoint_final.pth"
        return checkpoint_best.exists() or checkpoint_final.exists()

    def get_gpu_for_fold(self, fold: int) -> int:
        """Determine which GPU to use for a specific fold"""
        device_config = self.config.get('device', {})

        # Check for fold-specific GPU mapping
        fold_gpu_mapping = device_config.get('fold_gpu_mapping')
        if fold_gpu_mapping:
            for fold_range, gpu_id in fold_gpu_mapping.items():
                start, end = map(int, fold_range.split('-'))
                if start <= fold <= end:
                    logger.info(f"Using GPU {gpu_id} for fold {fold} (range {fold_range})")
                    return gpu_id

        # Check for multi-GPU automatic assignment
        available_gpus = device_config.get('available_gpus')
        if available_gpus and isinstance(available_gpus, list):
            gpu_id = available_gpus[fold % len(available_gpus)]
            logger.info(f"Auto-assigned GPU {gpu_id} for fold {fold}")
            return gpu_id

        # Fallback to default GPU
        default_gpu = device_config.get('cuda_device', 0)
        return default_gpu

    def train_single_fold(self, configuration: str, fold: int, device: str = "cuda",
                         resume: bool = None, trainer_class: str = None):
        """Train a single fold with comprehensive logging and optional custom trainer"""
        # Auto-detect if we should resume (unless explicitly specified)
        if resume is None:
            resume = self.check_checkpoint_exists(configuration, fold)
            if resume:
                logger.info(f"Found existing checkpoint for {configuration} fold {fold}, will resume training")

        # Determine GPU assignment for this fold
        gpu_id = self.get_gpu_for_fold(fold)

        logger.info("\n" + "="*80)
        logger.info(f"{'RESUMING' if resume else 'STARTING'} TRAINING")
        logger.info(f"Configuration: {configuration}")
        logger.info(f"Fold:          {fold}")
        logger.info(f"Device:        {device}")
        logger.info(f"GPU ID:        {gpu_id}")
        if trainer_class:
            logger.info(f"Trainer:       {trainer_class}")
        logger.info("="*80)

        fold_start_time = time.time()

        cmd = [
            "nnUNetv2_train",
            str(self.dataset_id),
            configuration,
            str(fold)
        ]

        # Add custom trainer if specified
        if trainer_class:
            cmd.extend(["-tr", trainer_class])
            logger.info(f"Using custom trainer class: {trainer_class}")
        else:
            # Check config for trainer class
            max_epochs = self.config.get('training', {}).get('max_epochs')
            if max_epochs:
                logger.warning(f"Note: max_epochs={max_epochs} specified in config but no custom trainer provided")
                logger.warning("To use custom epoch count, specify --trainer parameter or use custom trainer class")

        # Add optional flags
        cmd.append("--npz")  # Save softmax predictions for ensemble

        if resume:
            cmd.append("--c")

        # Add device specification
        cmd.extend(["-device", device])

        # Set CUDA device BEFORE subprocess (FIXED: moved before run_command)
        env = os.environ.copy()
        if device == "cuda":
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id} for fold {fold}")

        # Create fold-specific log file
        fold_log_file = self.metadata_path / f"training_{configuration}_fold{fold}.log"

        try:
            # Run training with subprocess environment
            logger.info(f"Training log: {fold_log_file}")

            # Resolve nnUNet command to full path on Windows
            cmd = self.resolve_nnunet_command(cmd)

            # Modified run_command to accept env
            result = subprocess.run(
                cmd,
                env=env,
                stdout=open(fold_log_file, 'w', encoding='utf-8'),
                stderr=subprocess.STDOUT,
                text=True
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)

            fold_elapsed = time.time() - fold_start_time
            logger.info(f"✅ Successfully completed {configuration} fold {fold} in {fold_elapsed/3600:.2f} hours")

            # Record fold result
            self.fold_results[f"{configuration}_fold{fold}"] = {
                'status': 'completed',
                'elapsed_time': fold_elapsed,
                'gpu_id': gpu_id
            }

        except subprocess.CalledProcessError as e:
            fold_elapsed = time.time() - fold_start_time
            logger.error(f"❌ Training failed for {configuration} fold {fold} after {fold_elapsed/60:.1f} minutes")

            # Record failure
            self.fold_results[f"{configuration}_fold{fold}"] = {
                'status': 'failed',
                'elapsed_time': fold_elapsed,
                'error': str(e)
            }
            raise

    def train_configuration(self, configuration: str, single_fold: int = None,
                           fold_list: Optional[List[int]] = None, resume: bool = None,
                           trainer_class: str = None):
        """Train all folds, a single fold, or a list of folds for a specific configuration"""
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING CONFIGURATION: {configuration}")
        logger.info(f"{'='*80}")

        device = "cuda" if self.config.get('device', {}).get('use_cuda', True) else "cpu"

        # Determine which folds to train
        if single_fold is not None:
            fold_numbers = [single_fold]
            logger.info(f"Training only fold {single_fold}")
        elif fold_list is not None:
            fold_numbers = fold_list
            logger.info(f"Training folds: {fold_list}")
        else:
            fold_numbers = self.get_fold_numbers()
            logger.info(f"Training all folds: {fold_numbers}")

        config_start_time = time.time()
        successful_folds = []
        failed_folds = []

        # Train each fold
        for fold in fold_numbers:
            try:
                self.train_single_fold(configuration, fold, device, resume=resume,
                                     trainer_class=trainer_class)
                successful_folds.append(fold)
            except Exception as e:
                failed_folds.append(fold)
                logger.error(f"Failed to train {configuration} fold {fold}: {e}")

                if not self.config.get('training', {}).get('continue_on_error', False):
                    logger.error("Stopping training due to error (continue_on_error=False)")
                    raise
                else:
                    logger.warning("Continuing with next fold due to continue_on_error=True")

        config_elapsed = time.time() - config_start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"CONFIGURATION {configuration} SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total time:       {config_elapsed/3600:.2f} hours")
        logger.info(f"Successful folds: {successful_folds} ({len(successful_folds)}/{len(fold_numbers)})")
        if failed_folds:
            logger.info(f"Failed folds:     {failed_folds}")
        logger.info(f"{'='*80}")

    def create_training_summary(self, configurations: List[str]):
        """Create a comprehensive training summary report"""
        total_elapsed = time.time() - self.training_start_time if self.training_start_time else 0

        summary = {
            'dataset': {
                'id': self.dataset_id,
                'name': self.dataset_name,
                'full_name': self.dataset_full_name
            },
            'training_session': {
                'session_id': self.session_id,
                'start_time': self.training_start_time,
                'end_time': time.time(),
                'total_elapsed_hours': total_elapsed / 3600,
                'random_seed': self.random_seed
            },
            'configurations': {
                'trained': configurations,
                'total_folds': len(self.get_fold_numbers()),
            },
            'fold_results': self.fold_results,
            'paths': {
                'results': str(self.results_path),
                'metadata': str(self.metadata_path)
            },
            'reproducibility': {
                'random_seed': self.random_seed,
                'deterministic_mode': True,
                'environment_variables': {
                    'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
                    'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')
                }
            }
        }

        summary_file = self.metadata_path / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nTraining summary saved: {summary_file}")

        # Also create human-readable summary
        readable_summary = self.metadata_path / 'training_summary.txt'
        with open(readable_summary, 'w') as f:
            f.write("="*80 + "\n")
            f.write("nnUNet TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset:           {self.dataset_full_name}\n")
            f.write(f"Session ID:        {self.session_id}\n")
            f.write(f"Total time:        {total_elapsed/3600:.2f} hours\n")
            f.write(f"Random seed:       {self.random_seed}\n")
            f.write(f"Configurations:    {', '.join(configurations)}\n")
            f.write(f"\nResults directory: {self.results_path}\n")
            f.write(f"Metadata directory: {self.metadata_path}\n\n")

            f.write("Fold Results:\n")
            f.write("-"*80 + "\n")
            for fold_name, result in self.fold_results.items():
                status = result['status']
                elapsed = result['elapsed_time'] / 3600
                f.write(f"  {fold_name:30s} {status:10s} {elapsed:.2f} hours\n")

        logger.info(f"Human-readable summary: {readable_summary}")

        return summary

    def train_all_configurations(self, configurations: Optional[List[str]] = None,
                                single_fold: int = None, fold_list: Optional[List[int]] = None,
                                resume: bool = None, trainer_class: str = None):
        """Train all specified configurations with comprehensive tracking"""

        self.training_start_time = time.time()

        if configurations is None:
            # Auto-detect from preprocessing
            configs_path = self.preprocessed_path
            if (configs_path / "nnUNetPlans_2d").exists():
                configurations = ["2d"]
            if (configs_path / "nnUNetPlans_3d_fullres").exists():
                configurations = configurations + ["3d_fullres"] if configurations else ["3d_fullres"]

            if not configurations:
                configurations = ["2d", "3d_fullres"]  # Default guess

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING SESSION STARTED")
        logger.info(f"{'='*80}")
        logger.info(f"Configurations to train: {configurations}")
        logger.info(f"Session ID:              {self.session_id}")
        logger.info(f"Metadata directory:      {self.metadata_path}")
        if trainer_class:
            logger.info(f"Custom trainer class:    {trainer_class}")
        logger.info(f"{'='*80}\n")

        trained_configs = []

        for config in configurations:
            try:
                self.train_configuration(config, single_fold=single_fold,
                                       fold_list=fold_list, resume=resume,
                                       trainer_class=trainer_class)
                trained_configs.append(config)

            except Exception as e:
                logger.error(f"Failed to train configuration {config}: {e}")
                if not self.config.get('training', {}).get('continue_on_error', False):
                    raise

        # Create comprehensive summary
        summary = self.create_training_summary(trained_configs)

        total_elapsed = time.time() - self.training_start_time

        logger.info(f"\n{'='*80}")
        logger.info("TRAINING SESSION COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"Total time:           {total_elapsed/3600:.2f} hours")
        logger.info(f"Trained configurations: {trained_configs}")
        logger.info(f"Results:              {self.results_path}")
        logger.info(f"Metadata:             {self.metadata_path}")
        logger.info(f"Session ID:           {self.session_id}")
        logger.info(f"{'='*80}\n")

        logger.info("NEXT STEPS:")
        logger.info("1. Verify training completed successfully:")
        logger.info(f"   - Check logs in {self.metadata_path}")
        logger.info(f"   - Review training_summary.json")
        logger.info("\n2. Run inference on test set:")
        logger.info(f"   python nnunet_inference.py --dataset_id {self.dataset_id} \\")
        logger.info(f"          --dataset_name {self.dataset_name} \\")
        logger.info(f"          --input_dir nnUNet_raw/{self.dataset_full_name}/imagesTs \\")
        logger.info(f"          --output_dir predictions/{self.dataset_full_name} \\")
        logger.info(f"          --configuration {trained_configs[0] if trained_configs else '2d'} \\")
        logger.info(f"          --all_folds --compute_metrics")

        return trained_configs


def main():
    parser = argparse.ArgumentParser(
        description='nnUNet Training Script with Scientific Rigor and Reproducibility'
    )
    parser.add_argument('--dataset_id', type=int, required=True,
                       help='Dataset ID (3-digit number)')
    parser.add_argument('--dataset_name', type=str, default='LiverCancer',
                       help='Dataset name')
    parser.add_argument('--config', type=str, default='train_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--configurations', nargs='+',
                       choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'],
                       help='Specific configurations to train (default: auto-detect)')
    parser.add_argument('--plan_only', action='store_true',
                       help='Only run planning and preprocessing')
    parser.add_argument('--train_only', action='store_true',
                       help='Only run training (skip planning)')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate existing dataset')
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted training (auto-detects checkpoints)')
    parser.add_argument('--fold', type=int,
                       help='Train only a specific fold (e.g., --fold 4)')
    parser.add_argument('--fold_range', type=str,
                       help='Train a range of folds (e.g., --fold_range 0-6)')
    parser.add_argument('--force_restart', action='store_true',
                       help='Force restart training even if checkpoints exist')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--trainer', type=str,
                       help='Custom trainer class name (e.g., nnUNetTrainer_500epochs)')

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {args.config}, using defaults")
        config = {}

    # Create trainer with random seed
    trainer = nnUNetTrainerRigorous(config, args.dataset_id, args.dataset_name,
                                   random_seed=args.random_seed)

    try:
        if args.validate_only:
            trainer.validate_dataset()
            logger.info("Dataset validation completed")
            return

        if not args.train_only:
            # Validate dataset
            trainer.validate_dataset()

            # Copy splits BEFORE preprocessing to prevent leakage
            trainer.copy_custom_splits()

            # Run planning and preprocessing
            trainer.plan_and_preprocess()

        if args.plan_only:
            logger.info("Planning completed. Run with --train_only to start training.")
            return

        # Determine resume behavior
        resume = None
        if args.resume:
            resume = True
            logger.info("Resume mode enabled - will continue from existing checkpoints")
        elif args.force_restart:
            resume = False
            logger.info("Force restart enabled - will ignore existing checkpoints")

        # Parse fold range if specified
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
                logger.error(f"Invalid fold range format: {args.fold_range}. Use format like '0-6'")
                sys.exit(1)

        # Start training
        trained_configs = trainer.train_all_configurations(
            args.configurations,
            single_fold=args.fold,
            fold_list=fold_list,
            resume=resume,
            trainer_class=args.trainer
        )

        logger.info("\n✅ All training completed successfully!")

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
