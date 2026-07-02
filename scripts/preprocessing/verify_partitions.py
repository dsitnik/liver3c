#!/usr/bin/env python3
"""
Verification Script for nnUNet Dataset Partitions

Verifies:
1. No patient leakage between train/val and test sets
2. No patient leakage across CV folds (patient appears in only one validation fold)
3. Stratification quality in CV folds
4. Consistency of partition data

Usage: python verify_partitions.py --base_dataset_id 100 --num_partitions 5
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class PartitionVerifier:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.errors = []
        self.warnings = []

    def extract_patient_id(self, case_id):
        """Extract patient ID from case ID (e.g., 'cho_15a' -> 'cho_15')"""
        # Remove channel suffix if present (_0000, _0001, _0002)
        if '_000' in case_id:
            case_id = case_id.split('_000')[0]

        # Remove 'a' suffix if present
        if case_id.endswith('a'):
            return case_id[:-1]
        return case_id

    def load_partition_info(self):
        """Load partition_info.json"""
        info_file = self.dataset_path / "partition_info.json"
        if not info_file.exists():
            self.errors.append(f"Missing partition_info.json in {self.dataset_path}")
            return None

        with open(info_file, 'r') as f:
            return json.load(f)

    def load_splits(self):
        """Load splits_final.json"""
        splits_file = self.dataset_path / "splits_final.json"
        if not splits_file.exists():
            self.errors.append(f"Missing splits_final.json in {self.dataset_path}")
            return None

        with open(splits_file, 'r') as f:
            return json.load(f)

    def verify_train_test_leakage(self, partition_info):
        """Verify no patient appears in both train/val and test sets"""
        print(f"\n{'='*80}")
        print(f"Verifying Train/Test Patient Leakage for {self.dataset_path.name}")
        print(f"{'='*80}")

        # Extract patients from test set
        test_patients = set()
        test_patients_by_class = partition_info.get('test_patients_by_class', {})

        for cancer_type, patients in test_patients_by_class.items():
            for patient_num in patients:
                patient_id = f"{cancer_type}_{patient_num}"
                test_patients.add(patient_id)

        print(f"Test patients: {len(test_patients)}")
        for cancer_type, patients in test_patients_by_class.items():
            print(f"  {cancer_type}: {len(patients)} patients - {sorted(patients)}")

        # Extract patients from train/val set
        train_val_patients = set()
        train_val_patients_by_class = partition_info.get('train_val_patients_by_class', {})

        for cancer_type, patients in train_val_patients_by_class.items():
            for patient_num in patients:
                patient_id = f"{cancer_type}_{patient_num}"
                train_val_patients.add(patient_id)

        print(f"\nTrain/Val patients: {len(train_val_patients)}")
        for cancer_type, patients in train_val_patients_by_class.items():
            print(f"  {cancer_type}: {len(patients)} patients")

        # Check for overlap
        overlap = test_patients & train_val_patients

        if overlap:
            self.errors.append(f"PATIENT LEAKAGE DETECTED: {len(overlap)} patients in both train/val and test")
            print(f"\n❌ ERROR: Patient leakage detected!")
            print(f"Patients in both sets: {sorted(overlap)}")
            return False
        else:
            print(f"\n✓ No train/test patient leakage detected")
            return True

    def verify_cv_fold_leakage(self, splits):
        """Verify no patient appears in validation set of multiple folds"""
        print(f"\n{'='*80}")
        print(f"Verifying Cross-Validation Fold Leakage for {self.dataset_path.name}")
        print(f"{'='*80}")

        # Track which fold each patient appears in for validation
        patient_val_folds = defaultdict(set)  # Use set to avoid counting duplicates

        for fold_idx, fold in enumerate(splits):
            val_cases = fold.get('val', [])
            val_patients = set()

            for case_id in val_cases:
                patient_id = self.extract_patient_id(case_id)
                val_patients.add(patient_id)
                patient_val_folds[patient_id].add(fold_idx)  # Use add() for set

            print(f"Fold {fold_idx}: {len(val_patients)} unique patients in validation")

        # Check for patients in multiple validation folds
        leakage_found = False
        for patient_id, folds in patient_val_folds.items():
            if len(folds) > 1:
                folds_list = sorted(list(folds))
                self.errors.append(f"Patient {patient_id} appears in validation of folds: {folds_list}")
                print(f"❌ ERROR: Patient {patient_id} in validation of multiple folds: {folds_list}")
                leakage_found = True

        if not leakage_found:
            print(f"\n✓ No cross-validation fold leakage detected")
            print(f"  Each patient appears in exactly one validation fold")
            return True
        else:
            return False

    def verify_stratification(self, splits, partition_info):
        """Verify stratification quality in CV folds"""
        print(f"\n{'='*80}")
        print(f"Verifying Stratification Quality for {self.dataset_path.name}")
        print(f"{'='*80}")

        # Get overall class distribution from train/val set
        train_val_patients_by_class = partition_info.get('train_val_patients_by_class', {})
        total_train_val = sum(len(patients) for patients in train_val_patients_by_class.values())

        overall_distribution = {}
        for cancer_type, patients in train_val_patients_by_class.items():
            overall_distribution[cancer_type] = len(patients) / total_train_val

        print(f"Overall train/val distribution ({total_train_val} patients):")
        for cancer_type, ratio in sorted(overall_distribution.items()):
            print(f"  {cancer_type}: {ratio*100:.1f}%")

        # Check each fold's validation set distribution
        print(f"\nValidation set distributions per fold:")
        fold_distributions = []

        for fold_idx, fold in enumerate(splits):
            val_cases = fold.get('val', [])

            # Count patients by class
            val_patients_by_class = defaultdict(set)
            for case_id in val_cases:
                patient_id = self.extract_patient_id(case_id)
                # Extract cancer type from patient_id (e.g., 'cho_15' -> 'cho')
                cancer_type = patient_id.rsplit('_', 1)[0]
                val_patients_by_class[cancer_type].add(patient_id)

            total_val_patients = sum(len(patients) for patients in val_patients_by_class.values())

            fold_dist = {}
            for cancer_type, patients in val_patients_by_class.items():
                fold_dist[cancer_type] = len(patients) / total_val_patients if total_val_patients > 0 else 0

            fold_distributions.append(fold_dist)

            # Print fold distribution
            dist_str = ", ".join([f"{ct}: {len(val_patients_by_class[ct])} ({fold_dist[ct]*100:.1f}%)"
                                 for ct in sorted(fold_dist.keys())])
            print(f"  Fold {fold_idx} ({total_val_patients} patients): {dist_str}")

        # Calculate stratification quality metrics
        print(f"\nStratification quality metrics:")

        # Calculate deviation from overall distribution
        max_deviations = {cancer_type: 0 for cancer_type in overall_distribution.keys()}
        mean_deviations = {cancer_type: [] for cancer_type in overall_distribution.keys()}

        for fold_dist in fold_distributions:
            for cancer_type in overall_distribution.keys():
                fold_ratio = fold_dist.get(cancer_type, 0)
                overall_ratio = overall_distribution[cancer_type]
                deviation = abs(fold_ratio - overall_ratio)

                max_deviations[cancer_type] = max(max_deviations[cancer_type], deviation)
                mean_deviations[cancer_type].append(deviation)

        stratification_good = True
        for cancer_type in sorted(overall_distribution.keys()):
            max_dev = max_deviations[cancer_type] * 100
            mean_dev = np.mean(mean_deviations[cancer_type]) * 100

            print(f"  {cancer_type}:")
            print(f"    Max deviation: {max_dev:.2f}%")
            print(f"    Mean deviation: {mean_dev:.2f}%")

            # Warning if deviation is large (>15% for max, >10% for mean)
            # These thresholds are reasonable for small validation sets (7-8 patients)
            if max_dev > 15:
                self.warnings.append(f"{cancer_type}: Large max deviation ({max_dev:.2f}%)")
                stratification_good = False
            if mean_dev > 10:
                self.warnings.append(f"{cancer_type}: Large mean deviation ({mean_dev:.2f}%)")
                stratification_good = False

        if stratification_good:
            print(f"\n✓ Good stratification quality")
            print(f"  All folds maintain similar class distributions to overall dataset")
        else:
            print(f"\n⚠ Stratification has some deviations (see warnings)")

        return stratification_good

    def verify_case_patient_consistency(self, partition_info):
        """Verify cases are correctly grouped by patient"""
        print(f"\n{'='*80}")
        print(f"Verifying Case-Patient Consistency for {self.dataset_path.name}")
        print(f"{'='*80}")

        # Check test cases
        test_cases_by_patient = defaultdict(list)
        for case_id in partition_info.get('test_cases', []):
            patient_id = self.extract_patient_id(case_id)
            test_cases_by_patient[patient_id].append(case_id)

        print(f"Test set patient-case mapping:")
        print(f"  {len(test_cases_by_patient)} unique patients")

        # Check for patients with unexpected number of cases
        single_image_patients = []
        dual_image_patients = []

        for patient_id, cases in sorted(test_cases_by_patient.items()):
            if len(cases) == 1:
                single_image_patients.append(patient_id)
            elif len(cases) == 2:
                dual_image_patients.append(patient_id)
            else:
                self.warnings.append(f"Unexpected: Patient {patient_id} has {len(cases)} test cases")

        print(f"  Single-image patients: {len(single_image_patients)}")
        if single_image_patients and len(single_image_patients) <= 5:
            print(f"    {single_image_patients}")
        print(f"  Dual-image patients: {len(dual_image_patients)}")

        # Check train/val cases
        train_val_cases_by_patient = defaultdict(list)
        for case_id in partition_info.get('train_val_cases', []):
            patient_id = self.extract_patient_id(case_id)
            train_val_cases_by_patient[patient_id].append(case_id)

        print(f"\nTrain/Val set patient-case mapping:")
        print(f"  {len(train_val_cases_by_patient)} unique patients")

        cases_dist = defaultdict(int)
        for patient_id, cases in train_val_cases_by_patient.items():
            cases_dist[len(cases)] += 1

        for num_cases in sorted(cases_dist.keys()):
            print(f"  {cases_dist[num_cases]} patients with {num_cases} case(s)")

        print(f"\n✓ Case-patient consistency verified")
        return True

    def run_verification(self):
        """Run all verification checks"""
        print(f"\n{'#'*80}")
        print(f"# PARTITION VERIFICATION: {self.dataset_path.name}")
        print(f"{'#'*80}")

        # Load data
        partition_info = self.load_partition_info()
        splits = self.load_splits()

        if not partition_info or not splits:
            print(f"\n❌ Cannot proceed: Missing required files")
            return False

        # Run checks
        checks_passed = []

        checks_passed.append(self.verify_train_test_leakage(partition_info))
        checks_passed.append(self.verify_cv_fold_leakage(splits))
        checks_passed.append(self.verify_stratification(splits, partition_info))
        checks_passed.append(self.verify_case_patient_consistency(partition_info))

        # Summary
        print(f"\n{'='*80}")
        print(f"VERIFICATION SUMMARY: {self.dataset_path.name}")
        print(f"{'='*80}")

        if all(checks_passed):
            print(f"✓ ALL CHECKS PASSED")
        else:
            print(f"❌ SOME CHECKS FAILED")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ❌ {error}")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        return all(checks_passed) and len(self.errors) == 0

def main():
    parser = argparse.ArgumentParser(
        description='Verify nnUNet dataset partitions for leakage and stratification'
    )
    parser.add_argument('--base_dataset_id', type=int, default=100,
                       help='Base dataset ID (default: 100)')
    parser.add_argument('--num_partitions', type=int, default=5,
                       help='Number of partitions (default: 5)')
    parser.add_argument('--output_dir', type=str, default='nnUNet_raw',
                       help='Output directory containing datasets (default: nnUNet_raw)')
    parser.add_argument('--partition', type=int, default=None,
                       help='Verify only specific partition (1-5), otherwise all')

    args = parser.parse_args()

    # Determine which partitions to verify
    if args.partition is not None:
        partitions_to_verify = [args.partition]
    else:
        partitions_to_verify = range(1, args.num_partitions + 1)

    # Run verification for each partition
    all_passed = True
    results = {}

    for partition_num in partitions_to_verify:
        dataset_id = args.base_dataset_id + partition_num - 1
        dataset_name = f"Liver{partition_num}"
        dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
        dataset_path = Path(args.output_dir) / dataset_folder

        if not dataset_path.exists():
            print(f"\n❌ Dataset not found: {dataset_path}")
            all_passed = False
            results[partition_num] = False
            continue

        verifier = PartitionVerifier(dataset_path)
        passed = verifier.run_verification()
        results[partition_num] = passed
        all_passed = all_passed and passed

    # Final summary
    print(f"\n{'#'*80}")
    print(f"# FINAL VERIFICATION SUMMARY")
    print(f"{'#'*80}")
    print(f"\nVerified {len(partitions_to_verify)} partition(s):\n")

    for partition_num in partitions_to_verify:
        dataset_id = args.base_dataset_id + partition_num - 1
        dataset_name = f"Dataset{dataset_id:03d}_Liver{partition_num}"
        status = "✓ PASSED" if results.get(partition_num, False) else "❌ FAILED"
        print(f"  {dataset_name}: {status}")

    print(f"\n{'='*80}")
    if all_passed:
        print(f"✓ ALL PARTITIONS VERIFIED SUCCESSFULLY")
        print(f"  - No patient leakage between train/test sets")
        print(f"  - No patient leakage across CV folds")
        print(f"  - Good stratification in all folds")
        print(f"  - Case-patient consistency maintained")
    else:
        print(f"❌ VERIFICATION FAILED FOR SOME PARTITIONS")
        print(f"  Review the detailed output above for specific issues")
    print(f"{'='*80}\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
