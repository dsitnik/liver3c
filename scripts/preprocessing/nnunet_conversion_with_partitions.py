#!/usr/bin/env python3
"""
nnUNet Dataset Conversion Script with Predefined Test Partitions
Creates 5 separate datasets based on test_partitions.json with stratified CV for train/val.

Usage: python nnunet_conversion_with_partitions.py --base_dataset_id 100 --partitions_file data/test_partitions.json
"""

import os
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_nnunet_folders(base_dir, dataset_id, dataset_name):
    """Create nnUNet folder structure"""
    dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_path = Path(base_dir) / dataset_folder

    # Create directories
    (dataset_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dataset_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labelsTs").mkdir(parents=True, exist_ok=True)  # For test labels (evaluation)

    logger.info(f"Created dataset folder: {dataset_path}")
    return dataset_path

def get_patient_case_id(cancer_type, patient_num, suffix=''):
    """Generate case ID from cancer type and patient number"""
    return f"{cancer_type}_{patient_num}{suffix}"

def find_patient_images(images_dir, cancer_type, patient_num):
    """Find all images for a patient (handles both single and dual images)"""
    images_dir = Path(images_dir)
    cases = []

    # Check for base image
    base_case = get_patient_case_id(cancer_type, patient_num)
    if (images_dir / f"{base_case}.npy").exists():
        cases.append(base_case)

    # Check for 'a' suffix image
    a_case = get_patient_case_id(cancer_type, patient_num, 'a')
    if (images_dir / f"{a_case}.npy").exists():
        cases.append(a_case)

    return cases

def load_partition_data(images_dir, labels_dir, partition_info, excluded_patients):
    """Load test and train/val cases for a specific partition"""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    test_cases = []
    test_patients_by_class = defaultdict(list)
    train_val_cases = []
    train_val_patients_by_class = defaultdict(list)
    patient_cases = defaultdict(list)
    case_info = {}

    # Process each cancer type
    for cancer_type in ['metastatic', 'hcc', 'cho']:
        # Get test patient numbers for this class
        test_patient_nums = partition_info.get(cancer_type, [])

        # Get excluded patient numbers for this class
        excluded_nums = excluded_patients.get(cancer_type, [])

        # Find all patients for this cancer type
        all_patient_nums = set()
        for image_file in images_dir.glob(f'{cancer_type}_*.npy'):
            case_id = image_file.stem
            # Extract patient number (remove 'a' suffix if present)
            patient_id = case_id.replace(f'{cancer_type}_', '')
            if patient_id.endswith('a'):
                patient_id = patient_id[:-1]
            try:
                patient_num = int(patient_id)
                all_patient_nums.add(patient_num)
            except ValueError:
                logger.warning(f"Could not parse patient number from {case_id}")
                continue

        # Process test patients
        for patient_num in test_patient_nums:
            if patient_num in excluded_nums:
                logger.info(f"Skipping excluded patient: {cancer_type}_{patient_num}")
                continue

            cases = find_patient_images(images_dir, cancer_type, patient_num)
            if not cases:
                logger.warning(f"No images found for test patient {cancer_type}_{patient_num}")
                continue

            # Verify labels exist
            valid_cases = []
            for case_id in cases:
                label_file = labels_dir / f"{case_id}.npy"
                if not label_file.exists():
                    logger.warning(f"Missing label for {case_id}, skipping...")
                    continue
                valid_cases.append(case_id)

            if valid_cases:
                test_cases.extend(valid_cases)
                test_patients_by_class[cancer_type].append(patient_num)
                patient_cases[f"{cancer_type}_{patient_num}"] = valid_cases

                # Load label info for first case
                try:
                    label_file = labels_dir / f"{valid_cases[0]}.npy"
                    label = np.load(label_file)
                    class_counts = label.sum(axis=(0, 1))
                    tumor_counts = class_counts[1:]

                    for case_id in valid_cases:
                        case_info[case_id] = {
                            'patient_id': f"{cancer_type}_{patient_num}",
                            'class': cancer_type,
                            'tumor_pixels': int(tumor_counts.sum()),
                            'total_pixels': int(label.shape[0] * label.shape[1])
                        }
                except Exception as e:
                    logger.error(f"Error loading label for {cancer_type}_{patient_num}: {str(e)}")

        # Process train/val patients (all patients not in test and not excluded)
        train_val_patient_nums = all_patient_nums - set(test_patient_nums) - set(excluded_nums)

        for patient_num in train_val_patient_nums:
            cases = find_patient_images(images_dir, cancer_type, patient_num)
            if not cases:
                continue

            # Verify labels exist
            valid_cases = []
            for case_id in cases:
                label_file = labels_dir / f"{case_id}.npy"
                if not label_file.exists():
                    logger.warning(f"Missing label for {case_id}, skipping...")
                    continue
                valid_cases.append(case_id)

            if valid_cases:
                train_val_cases.extend(valid_cases)
                train_val_patients_by_class[cancer_type].append(patient_num)
                patient_cases[f"{cancer_type}_{patient_num}"] = valid_cases

                # Load label info for first case
                try:
                    label_file = labels_dir / f"{valid_cases[0]}.npy"
                    label = np.load(label_file)
                    class_counts = label.sum(axis=(0, 1))
                    tumor_counts = class_counts[1:]

                    for case_id in valid_cases:
                        case_info[case_id] = {
                            'patient_id': f"{cancer_type}_{patient_num}",
                            'class': cancer_type,
                            'tumor_pixels': int(tumor_counts.sum()),
                            'total_pixels': int(label.shape[0] * label.shape[1])
                        }
                except Exception as e:
                    logger.error(f"Error loading label for {cancer_type}_{patient_num}: {str(e)}")

    # Log statistics
    logger.info(f"TEST SET:")
    total_test_patients = 0
    for cancer_type, patients in test_patients_by_class.items():
        case_count = sum(len(patient_cases[f"{cancer_type}_{p}"]) for p in patients)
        logger.info(f"  {cancer_type}: {len(patients)} patients ({case_count} cases) - {sorted(patients)}")
        total_test_patients += len(patients)

    logger.info(f"TRAIN/VAL SET:")
    total_train_patients = 0
    for cancer_type, patients in train_val_patients_by_class.items():
        case_count = sum(len(patient_cases[f"{cancer_type}_{p}"]) for p in patients)
        logger.info(f"  {cancer_type}: {len(patients)} patients ({case_count} cases)")
        total_train_patients += len(patients)

    logger.info(f"Total: {total_train_patients} train/val patients ({len(train_val_cases)} cases), "
               f"{total_test_patients} test patients ({len(test_cases)} cases)")

    return train_val_cases, test_cases, train_val_patients_by_class, test_patients_by_class, patient_cases, case_info

def create_stratified_patient_cv_folds(train_val_patients_by_class, patient_cases, n_folds=10):
    """Create STRATIFIED cross-validation folds based on patients to ensure balanced class distribution"""

    # Create patient list with class labels
    patients = []  # Will store full patient IDs like "cho_15"
    labels = []

    for class_name, class_patients in train_val_patients_by_class.items():
        for patient_num in class_patients:
            # Create full patient ID
            patient_id = f"{class_name}_{patient_num}"
            patients.append(patient_id)
            labels.append(class_name)

    logger.info(f"\nCreating {n_folds}-fold STRATIFIED CV with {len(patients)} patients:")
    label_counts = {}
    for label in set(labels):
        count = labels.count(label)
        label_counts[label] = count
        logger.info(f"  {label}: {count} patients ({count/len(patients)*100:.1f}%)")

    # Stratified K-fold: ensures each fold has similar class distribution
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patients, labels)):
        # Get patient IDs (full IDs like "cho_15")
        train_patients = [patients[i] for i in train_idx]
        val_patients = [patients[i] for i in val_idx]

        # Convert patients to cases
        train_fold = []
        val_fold = []

        for patient_id in train_patients:
            cases = patient_cases[patient_id]
            train_fold.extend(cases)

        for patient_id in val_patients:
            cases = patient_cases[patient_id]
            val_fold.extend(cases)

        folds.append({
            "train": train_fold,
            "val": val_fold
        })

        # Log with class breakdown for validation set
        val_classes = [labels[i] for i in val_idx]
        class_counts = {}
        for cls in set(val_classes):
            class_counts[cls] = val_classes.count(cls)

        class_dist_str = ", ".join([f"{cls}: {count} ({count/len(val_patients)*100:.1f}%)"
                                    for cls, count in sorted(class_counts.items())])
        logger.info(f"Fold {fold_idx}: {len(train_fold)} train cases, {len(val_fold)} val cases "
                   f"({len(train_patients)} train patients, {len(val_patients)} val patients) "
                   f"- Val: [{class_dist_str}]")

    return folds

def convert_images_to_nnunet_format(images_dir, labels_dir, output_path, file_list, is_test=False):
    """Convert .npy images and labels to nnUNet format"""
    if is_test:
        images_path = output_path / "imagesTs"
        labels_path = output_path / "labelsTs"
    else:
        images_path = output_path / "imagesTr"
        labels_path = output_path / "labelsTr"

    converted_files = []

    for case_id in file_list:
        # Load image and label
        image_path = Path(images_dir) / f"{case_id}.npy"
        label_path = Path(labels_dir) / f"{case_id}.npy"

        if not image_path.exists() or not label_path.exists():
            logger.warning(f"Missing files for case {case_id}, skipping...")
            continue

        try:
            # Load data
            image = np.load(image_path)  # Shape: (H, W, 3)
            label = np.load(label_path)  # Shape: (H, W, 4) - one-hot

            # Convert one-hot labels to class indices
            label_indices = np.argmax(label, axis=2)  # Shape: (H, W)

            # Save each channel as a separate file
            for channel in range(3):
                channel_data = image[:, :, channel]

                # Convert to PNG format
                if channel_data.max() <= 1.0:
                    channel_data = (channel_data * 255).astype(np.uint8)
                else:
                    channel_data = np.clip(channel_data, 0, 255).astype(np.uint8)

                # Save as PNG
                output_file = images_path / f"{case_id}_{channel:04d}.png"
                Image.fromarray(channel_data, mode='L').save(output_file)

            # Save label as PNG
            label_output = labels_path / f"{case_id}.png"
            label_indices_uint8 = label_indices.astype(np.uint8)
            Image.fromarray(label_indices_uint8, mode='L').save(label_output)

            converted_files.append(case_id)

        except Exception as e:
            logger.error(f"Error processing {case_id}: {str(e)}")
            continue

    logger.info(f"Converted {len(converted_files)} {'test' if is_test else 'training'} cases")
    return converted_files

def create_dataset_json(output_path, dataset_name, num_training, num_test):
    """Create dataset.json file required by nnUNet"""
    dataset_json = {
        "channel_names": {
            "0": "channel_0",
            "1": "channel_1",
            "2": "channel_2"
        },
        "labels": {
            "background": 0,
            "metastatic": 1,
            "hcc": 2,
            "cho": 3
        },
        "numTraining": num_training,
        "numTest": num_test,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }

    # Save dataset.json
    with open(output_path / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)

    logger.info(f"Created dataset.json with {num_training} training and {num_test} test cases")
    return dataset_json

def save_splits_file(output_path, cv_folds):
    """Save splits_final.json for nnUNet"""
    splits_file = output_path / "splits_final.json"
    with open(splits_file, 'w') as f:
        json.dump(cv_folds, f, indent=2)

    logger.info(f"Created splits_final.json with {len(cv_folds)} folds")

def save_partition_info(output_path, partition_num, test_cases, train_val_cases,
                       test_patients_by_class, train_val_patients_by_class):
    """Save information about the partition for future reference"""
    partition_info = {
        "partition_number": partition_num,
        "test_cases": test_cases,
        "train_val_cases": train_val_cases,
        "num_test": len(test_cases),
        "num_train_val": len(train_val_cases),
        "test_patients_by_class": {k: sorted(v) for k, v in test_patients_by_class.items()},
        "train_val_patients_by_class": {k: sorted(v) for k, v in train_val_patients_by_class.items()},
        "total_test_patients": sum(len(v) for v in test_patients_by_class.values()),
        "total_train_val_patients": sum(len(v) for v in train_val_patients_by_class.values()),
        "description": f"Partition {partition_num} - Predefined test set from test_partitions.json. CV uses STRATIFIED K-fold for balanced class distribution."
    }

    with open(output_path / "partition_info.json", 'w') as f:
        json.dump(partition_info, f, indent=2)

    logger.info(f"Saved partition information to partition_info.json")

def create_evaluation_script(output_path, dataset_id, dataset_name, partition_num):
    """Create a script for evaluating on the test set"""
    script_content = f'''#!/bin/bash
"""
nnUNet Test Set Evaluation Script for {dataset_name} (Partition {partition_num})
Evaluates trained models on the hold-out test set using BEST checkpoints

Usage: bash evaluate_test_set.sh [--use-final]
       --use-final: Use final checkpoint instead of best (not recommended)
"""

set -e

DATASET_ID={dataset_id}
DATASET_NAME="{dataset_name}"
DATASET_FULL="Dataset{dataset_id:03d}_{dataset_name}"

# Check if --use-final flag is provided
USE_FINAL=false
if [[ "$1" == "--use-final" ]]; then
    USE_FINAL=true
    echo "WARNING: Using final checkpoint instead of best checkpoint"
    CHECKPOINT="checkpoint_final"
    CHECKPOINT_DESC="final"
else
    echo "Using best checkpoint for evaluation (recommended)"
    CHECKPOINT="checkpoint_best"
    CHECKPOINT_DESC="best"
fi

echo "Evaluating nnUNet models on test set for $DATASET_FULL (Partition {partition_num})"
echo "Checkpoint: $CHECKPOINT"

# Set environment variables
export nnUNet_raw="$PWD/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnUNet_results"

# Create output directory for predictions
OUTPUT_DIR="$PWD/test_predictions_partition{partition_num}_$CHECKPOINT_DESC"
mkdir -p $OUTPUT_DIR

echo "Running inference on test set with $CHECKPOINT_DESC checkpoint..."

# Evaluate 2d configuration if it exists
if [ -d "$nnUNet_results/$DATASET_FULL/nnUNetTrainer__nnUNetPlans__2d" ]; then
    echo "Evaluating 2d configuration..."
    nnUNetv2_predict \\
        -i "$nnUNet_raw/$DATASET_FULL/imagesTs" \\
        -o "$OUTPUT_DIR/2d" \\
        -d $DATASET_ID \\
        -c 2d \\
        -f all \\
        -chk $CHECKPOINT \\
        --save_probabilities
fi

# Evaluate 3d_fullres configuration if it exists
if [ -d "$nnUNet_results/$DATASET_FULL/nnUNetTrainer__nnUNetPlans__3d_fullres" ]; then
    echo "Evaluating 3d_fullres configuration..."
    nnUNetv2_predict \\
        -i "$nnUNet_raw/$DATASET_FULL/imagesTs" \\
        -o "$OUTPUT_DIR/3d_fullres" \\
        -d $DATASET_ID \\
        -c 3d_fullres \\
        -f all \\
        -chk $CHECKPOINT \\
        --save_probabilities
fi

echo "Test set evaluation completed!"
echo "Predictions saved to: $OUTPUT_DIR"
echo ""
echo "To compute metrics, run:"
echo "python compute_test_metrics.py --predictions $OUTPUT_DIR/2d --labels $nnUNet_raw/$DATASET_FULL/labelsTs --output test_metrics_partition{partition_num}_${{CHECKPOINT_DESC}}_2d.txt"
echo "python compute_test_metrics.py --predictions $OUTPUT_DIR/3d_fullres --labels $nnUNet_raw/$DATASET_FULL/labelsTs --output test_metrics_partition{partition_num}_${{CHECKPOINT_DESC}}_3d.txt"
'''

    script_path = output_path / "evaluate_test_set.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # On Windows, we don't set execute permissions
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    logger.info(f"Created evaluation script: {script_path}")

def process_partition(partition_num, partition_data, excluded_patients, args):
    """Process a single partition and create corresponding dataset"""
    dataset_id = args.base_dataset_id + partition_num - 1
    dataset_name = f"{args.dataset_base_name}{partition_num}"

    logger.info("\n" + "="*80)
    logger.info(f"Processing Partition {partition_num}: Dataset{dataset_id:03d}_{dataset_name}")
    logger.info("="*80)

    # Create nnUNet folder structure
    dataset_path = create_nnunet_folders(args.output_dir, dataset_id, dataset_name)

    # Load partition data
    train_val_cases, test_cases, train_val_patients_by_class, test_patients_by_class, \
        patient_cases, case_info = load_partition_data(
            args.images_dir, args.labels_dir, partition_data, excluded_patients
        )

    if not train_val_cases or not test_cases:
        logger.error(f"Partition {partition_num} has no train/val or test cases! Skipping...")
        return False

    # Create stratified CV folds
    cv_folds = create_stratified_patient_cv_folds(
        train_val_patients_by_class, patient_cases, n_folds=args.n_folds
    )

    # Convert training/validation images
    logger.info("\nConverting training/validation images to nnUNet format...")
    converted_train = convert_images_to_nnunet_format(
        args.images_dir, args.labels_dir, dataset_path, train_val_cases, is_test=False
    )

    # Convert test images
    logger.info("Converting test images to nnUNet format...")
    converted_test = convert_images_to_nnunet_format(
        args.images_dir, args.labels_dir, dataset_path, test_cases, is_test=True
    )

    # Create dataset.json
    create_dataset_json(dataset_path, dataset_name, len(converted_train), len(converted_test))

    # Save splits_final.json for cross-validation
    save_splits_file(dataset_path, cv_folds)

    # Save partition information
    save_partition_info(dataset_path, partition_num, test_cases, train_val_cases,
                       test_patients_by_class, train_val_patients_by_class)

    # Create evaluation script
    create_evaluation_script(dataset_path, dataset_id, dataset_name, partition_num)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info(f"Partition {partition_num} completed successfully!")
    logger.info("="*60)
    logger.info(f"Dataset location: {dataset_path}")
    logger.info(f"Training/Validation cases: {len(converted_train)}")
    logger.info(f"Hold-out test cases: {len(converted_test)}")
    logger.info(f"CV method: STRATIFIED K-fold (balanced class distribution)")
    logger.info("\nTest set composition:")
    for class_name in ['metastatic', 'hcc', 'cho']:
        if class_name in test_patients_by_class:
            patients = sorted(test_patients_by_class[class_name])
            case_count = sum(len(patient_cases[f"{class_name}_{p}"]) for p in patients)
            logger.info(f"  {class_name}: {len(patients)} patients ({case_count} cases) - {patients}")

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Convert dataset to nnUNet format using predefined test partitions'
    )
    parser.add_argument('--base_dataset_id', type=int, default=100,
                       help='Base dataset ID (default: 100, creates 100-104)')
    parser.add_argument('--dataset_base_name', type=str, default='Liver',
                       help='Base dataset name (default: Liver, creates Liver1-Liver5)')
    parser.add_argument('--partitions_file', type=str, required=True,
                       help='Path to test_partitions.json file')
    parser.add_argument('--images_dir', type=str, default='data/images',
                       help='Directory containing input images')
    parser.add_argument('--labels_dir', type=str, default='data/labels',
                       help='Directory containing label images')
    parser.add_argument('--output_dir', type=str, default='nnUNet_raw',
                       help='Output directory for nnUNet datasets')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of cross-validation folds (default: 10)')
    parser.add_argument('--partition', type=int, default=None,
                       help='Process only specific partition (1-5), otherwise all')

    args = parser.parse_args()

    # Load partitions file
    logger.info(f"Loading partitions from {args.partitions_file}")
    with open(args.partitions_file, 'r') as f:
        partitions_data = json.load(f)

    excluded_patients = partitions_data.get('excluded_from_train', {})
    test_partitions = partitions_data.get('test_partitions', [])

    logger.info(f"Found {len(test_partitions)} partitions")
    if excluded_patients:
        logger.info("Excluded patients:")
        for cancer_type, patient_list in excluded_patients.items():
            if patient_list:
                logger.info(f"  {cancer_type}: {patient_list}")

    # Process partitions
    if args.partition is not None:
        # Process single partition
        if args.partition < 1 or args.partition > len(test_partitions):
            logger.error(f"Invalid partition number {args.partition}. Must be 1-{len(test_partitions)}")
            return

        partition = test_partitions[args.partition - 1]
        success = process_partition(args.partition, partition, excluded_patients, args)

        if success:
            logger.info(f"\n{'='*80}")
            logger.info(f"Successfully created Dataset{args.base_dataset_id + args.partition - 1:03d}_{args.dataset_base_name}{args.partition}")
            logger.info(f"{'='*80}")
    else:
        # Process all partitions
        successful_partitions = []
        for partition in test_partitions:
            partition_num = partition['partition']
            success = process_partition(partition_num, partition, excluded_patients, args)
            if success:
                successful_partitions.append(partition_num)

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("ALL PARTITIONS PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Successfully created {len(successful_partitions)} datasets:")
        for p_num in successful_partitions:
            dataset_id = args.base_dataset_id + p_num - 1
            dataset_name = f"{args.dataset_base_name}{p_num}"
            logger.info(f"  - Dataset{dataset_id:03d}_{dataset_name}")

        logger.info("\nNext steps:")
        logger.info("1. Set nnUNet environment variables:")
        logger.info(f"   export nnUNet_raw={Path(args.output_dir).absolute()}")
        logger.info(f"   export nnUNet_preprocessed=$PWD/nnUNet_preprocessed")
        logger.info(f"   export nnUNet_results=$PWD/nnUNet_results")
        logger.info("\n2. Train each partition:")
        for p_num in successful_partitions:
            dataset_id = args.base_dataset_id + p_num - 1
            logger.info(f"   nnUNetv2_train {dataset_id} 2d all")

if __name__ == "__main__":
    main()
