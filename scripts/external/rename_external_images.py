#!/usr/bin/env python3
"""
Rename external images to standardized format for diagnosis classification.

Naming rules:
  CHOL_*_sn_1.jpg        -> cho_*.jpg
  COLON_MA_intra_*_patient_N_sn_1.jpg -> metastatic_intra_N.jpg
  COLON_MA_*_sn_1.jpg    -> metastatic_*.jpg
  HCC_*_sn_1.jpg         -> hcc_*.jpg

All names lowercased, _sn_1 suffix stripped.
Produces a CSV mapping file for traceability.

Usage:
    python rename_external_images.py
    python rename_external_images.py --input_dir data/external_images_raw --output_dir data/external_images
"""

import argparse
import csv
import re
import shutil
from pathlib import Path


def rename_file(original_name: str) -> str:
    """Apply renaming rules to a single filename."""
    name = original_name

    # Strip _sn_1 before extension
    name = re.sub(r'_sn_1\.jpg$', '.jpg', name, flags=re.IGNORECASE)

    stem = Path(name).stem
    ext = Path(name).suffix  # .jpg

    # COLON_MA_intra_{num}_patient_{patient_id} -> metastatic_intra_{patient_id}
    m = re.match(r'COLON_MA_intra_\d+_patient_(\d+)', stem, re.IGNORECASE)
    if m:
        patient_id = m.group(1)
        return f"metastatic_intra_{patient_id}{ext}".lower()

    # COLON_MA_{id} -> metastatic_{id}
    m = re.match(r'COLON_MA_(.+)', stem, re.IGNORECASE)
    if m:
        remainder = m.group(1)
        return f"metastatic_{remainder}{ext}".lower()

    # CHOL_{id} -> cho_{id}
    m = re.match(r'CHOL_(.+)', stem, re.IGNORECASE)
    if m:
        remainder = m.group(1)
        return f"cho_{remainder}{ext}".lower()

    # HCC_{id} -> hcc_{id}
    m = re.match(r'HCC_(.+)', stem, re.IGNORECASE)
    if m:
        remainder = m.group(1)
        return f"hcc_{remainder}{ext}".lower()

    # Fallback: just lowercase
    return name.lower()


def main():
    parser = argparse.ArgumentParser(description='Rename external images to standardized format')
    parser.add_argument('--input_dir', type=str, default='data/external_images_raw',
                        help='Source directory with original images')
    parser.add_argument('--output_dir', type=str, default='data/external_images',
                        help='Destination directory for renamed images')
    parser.add_argument('--mapping_file', type=str, default='data/external_images_mapping.csv',
                        help='Output CSV mapping file')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print mapping without copying files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mapping_file = Path(args.mapping_file)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    image_files = sorted(
        {f for f in input_dir.glob("*.jpg")} |
        {f for f in input_dir.glob("*.JPG")}
    )
    if not image_files:
        print(f"Error: No JPG images found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = []
    seen_names = {}

    for img_file in image_files:
        original_name = img_file.name
        renamed = rename_file(original_name)

        # Check for collisions
        if renamed in seen_names:
            print(f"WARNING: Name collision! {original_name} -> {renamed} "
                  f"(already used by {seen_names[renamed]})")
            continue

        seen_names[renamed] = original_name
        mapping.append((original_name, renamed))

        if not args.dry_run:
            shutil.copy2(img_file, output_dir / renamed)

    # Write CSV mapping
    with open(mapping_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_name', 'renamed'])
        for original_name, renamed in mapping:
            writer.writerow([original_name, renamed])

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Renamed {len(mapping)} images")
    print(f"Mapping saved to: {mapping_file}")

    if not args.dry_run:
        print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
