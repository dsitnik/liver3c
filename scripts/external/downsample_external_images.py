#!/usr/bin/env python3
"""
Downsample 400x magnification external images to match training resolution.

Only affects images with '400' in the filename (2160x3840).
Non-400x images are left untouched.

Target resolutions (from training data):
  cho_400_*  -> 884 x 1124
  hcc_400_*  -> 883 x 1224

Usage:
    python downsample_external_images.py
    python downsample_external_images.py --dry_run
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

# Target resolutions per class (from training data, verified across all 5 datasets)
TARGET_RESOLUTIONS = {
    'cho': (884, 1124),   # (height, width)
    'hcc': (883, 1224),
}


def main():
    parser = argparse.ArgumentParser(
        description='Downsample 400x magnification images to match training resolution')
    parser.add_argument('--input_dir', type=str, default='data/external_images',
                        help='Directory with external images (default: data/external_images)')
    parser.add_argument('--log_file', type=str, default='data/external_images_downsample_log.csv',
                        help='CSV log of changes')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print changes without modifying files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        return

    # Only 400x images
    image_files = sorted(
        f for f in input_dir.glob("*.jpg") if '400' in f.stem
    )

    if not image_files:
        print("No 400x images found")
        return

    log_rows = []

    for img_path in image_files:
        lower = img_path.stem.lower()
        if lower.startswith('cho'):
            cls = 'cho'
        elif lower.startswith('hcc'):
            cls = 'hcc'
        else:
            print(f"  SKIP (no target for class): {img_path.name}")
            continue

        target_h, target_w = TARGET_RESOLUTIONS[cls]
        img = Image.open(img_path)
        orig_w, orig_h = img.size

        print(f"  {img_path.name}: {orig_h}x{orig_w} -> {target_h}x{target_w}")

        if not args.dry_run:
            img_resized = img.resize((target_w, target_h), Image.BICUBIC)
            img_resized.save(img_path, quality=95)

        log_rows.append((img_path.name, cls, f"{orig_h}x{orig_w}",
                         f"{target_h}x{target_w}"))

    if not args.dry_run and log_rows:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'class', 'original_resolution',
                             'target_resolution'])
            for row in log_rows:
                writer.writerow(row)
        print(f"\nLog saved to: {log_path}")

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{prefix}Downsampled {len(log_rows)} images")


if __name__ == "__main__":
    main()
