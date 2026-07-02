#!/usr/bin/env python3
"""
Package the input data for the three_types release as a single hosted archive.

Produces ``HandE-Liver3C_data.zip`` containing the raw dataset and the converted
nnU-Net datasets, with archive paths preserved relative to the project root so
that ``download_and_extract.py`` restores them into the exact locations the
pipeline scripts expect (``data/``, ``nnUNet_raw/``).

Run this from a full project checkout (the one that still holds the heavy data):

    python release_tools/package_data.py \
        --source /path/to/three_types \
        --output /path/to/three_types_release_assets

It prints the archive size and SHA256 -- copy the SHA256 into
``download_and_extract.py`` (``DATA_SHA256``) so end users can verify the download.
"""
import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

# Paths to include, relative to the source project root.
INCLUDE_DIRS = [
    "data/images",
    "data/labels",
    "data/external_images",
    "data/external_images_raw",
    "nnUNet_raw",
]
INCLUDE_FILES = [
    "data/test_partitions.json",
    "data/external_images_mapping.csv",
]
# Tiny per-dataset preprocessing artifacts. classification_inference.py reads
# nnUNet_preprocessed/<dataset>/dataset_fingerprint.json for intensity
# normalization, so these must be restored even though the bulk of
# nnUNet_preprocessed/ (cached training arrays) is intentionally excluded.
INCLUDE_GLOBS = [
    "nnUNet_preprocessed/*/dataset_fingerprint.json",
    "nnUNet_preprocessed/*/nnUNetPlans.json",
    "nnUNet_preprocessed/*/dataset.json",
]
# Never include these even if encountered inside the dirs above.
SKIP_EXTS = {".zip", ".pyc"}
SKIP_NAMES = {"__pycache__", ".DS_Store", "Thumbs.db"}


def iter_files(source: Path):
    for rel in INCLUDE_FILES:
        p = source / rel
        if p.is_file():
            yield p, rel
    for pattern in INCLUDE_GLOBS:
        for p in sorted(source.glob(pattern)):
            if p.is_file():
                yield p, str(p.relative_to(source)).replace(os.sep, "/")
    for rel_dir in INCLUDE_DIRS:
        base = source / rel_dir
        if not base.exists():
            print(f"  WARN: missing {rel_dir} (skipped)", file=sys.stderr)
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
            root_path = Path(root)
            for f in files:
                if Path(f).suffix.lower() in SKIP_EXTS or f in SKIP_NAMES:
                    continue
                src = root_path / f
                yield src, str(src.relative_to(source)).replace(os.sep, "/")


def write_zip(source: Path, out_zip: Path) -> int:
    count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arcname in iter_files(source):
            try:
                zf.write(src, arcname=arcname)
            except ValueError:
                # Pre-1980 timestamps: rewrite ZipInfo manually.
                zi = zipfile.ZipInfo(filename=arcname, date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                with open(src, "rb") as fp:
                    zf.writestr(zi, fp.read())
            count += 1
            if count % 200 == 0:
                print(f"  added {count} files...")
    return count


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path.cwd(),
                    help="Full project checkout holding data/ and nnUNet_raw/ (default: cwd)")
    ap.add_argument("--output", type=Path, default=Path.cwd(),
                    help="Directory to write HandE-Liver3C_data.zip into (default: cwd)")
    args = ap.parse_args()

    source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    out_zip = (args.output / "HandE-Liver3C_data.zip").resolve()
    if out_zip.exists():
        out_zip.unlink()

    print(f"Source: {source}")
    print(f"Output: {out_zip}")
    n = write_zip(source, out_zip)

    size_gb = out_zip.stat().st_size / (1024 ** 3)
    digest = sha256(out_zip)
    print("\nDone.")
    print(f"  Files:  {n}")
    print(f"  Size:   {size_gb:.2f} GB")
    print(f"  SHA256: {digest}")
    print("\n--> Paste this SHA256 into release_tools/download_and_extract.py (DATA_SHA256)")


if __name__ == "__main__":
    main()
