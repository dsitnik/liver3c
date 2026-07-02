#!/usr/bin/env python3
"""
Download and extract the hosted data / weights archives for the three_types
release into the correct project locations.

The archives are hosted in the FULIR / IRB repository (DOI object irb:896,
https://data.fulir.irb.hr/en/object/irb:896) and the direct download URLs are
already configured below; you can override them with --data-zip/--weights-zip to
use local copies. The archives unpack relative to the project root, restoring:

    HandE-Liver3C_data.zip    -> data/ , nnUNet_raw/
    HandE-Liver3C_weights.zip -> nnUNet_results/ , classification_results/

Examples
--------
    # Download both from the hosted URLs into the current project folder:
    python release_tools/download_and_extract.py

    # Use already-downloaded local zips instead of fetching:
    python release_tools/download_and_extract.py \
        --data-zip ./HandE-Liver3C_data.zip --weights-zip ./HandE-Liver3C_weights.zip

    # Only restore the data (skip the large weights archive):
    python release_tools/download_and_extract.py --skip-weights
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# --- Hosted archive locations (FULIR / IRB repository, DOI object irb:896) ----
# Landing page: https://data.fulir.irb.hr/en/object/irb:896
DATA_URL = "https://data.fulir.irb.hr/data/HandE-Liver3C/HandE-Liver3C_data.zip"        # HandE-Liver3C_data.zip
WEIGHTS_URL = "https://data.fulir.irb.hr/data/HandE-Liver3C/HandE-Liver3C_weights.zip"  # HandE-Liver3C_weights.zip
DATA_SHA256 = "231ba6fc5d2033ba341c75a96018767738864b15054189ccd529b5444f642ea7"  # HandE-Liver3C_data.zip (2.43 GB)
WEIGHTS_SHA256 = "6437664fbedc11aecb99d3f4770f93ad605fa38162326d6dc1afc104a130e87f"  # HandE-Liver3C_weights.zip (all folds, 225.92 GB)
# -----------------------------------------------------------------------------


# Repository root, derived from this script's own location (release_tools/ -> repo root).
# Used as the default extract target so the archives ALWAYS restore into the project root,
# regardless of which directory you launch this script from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 / total_size)
        sys.stdout.write(f"\r  downloading... {pct:5.1f}%")
        sys.stdout.flush()


def download(url: str, dest: Path):
    if url == "<FILL_ME>":
        raise SystemExit(
            f"URL not configured. Edit DATA_URL/WEIGHTS_URL in {Path(__file__).name} "
            "or pass --data-zip/--weights-zip with a local file.")
    print(f"  GET {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def verify(path: Path, expected: str):
    if not expected or expected == "<FILL_ME>":
        print("  (no checksum configured; skipping verification)")
        return
    print("  verifying SHA256...")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"  CHECKSUM MISMATCH for {path.name}\n"
                         f"    expected {expected}\n    actual   {actual}")
    print("  checksum OK")


def extract(zip_path: Path, target: Path):
    print(f"  extracting {zip_path.name} -> {target}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    print("  done")


def handle(name, url, expected, local, target, work_dir):
    print(f"\n=== {name} ===")
    if local:
        zip_path = Path(local).resolve()
        if not zip_path.is_file():
            raise SystemExit(f"  local zip not found: {zip_path}")
    else:
        zip_path = work_dir / f"{name}.zip"
        download(url, zip_path)
    verify(zip_path, expected)
    extract(zip_path, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, default=PROJECT_ROOT,
                    help="Project root to extract into (default: the repository root, "
                         "auto-detected from this script's location)")
    ap.add_argument("--data-zip", type=str, default=None,
                    help="Use this local HandE-Liver3C_data.zip instead of downloading")
    ap.add_argument("--weights-zip", type=str, default=None,
                    help="Use this local HandE-Liver3C_weights.zip instead of downloading")
    ap.add_argument("--skip-data", action="store_true", help="Do not restore data")
    ap.add_argument("--skip-weights", action="store_true", help="Do not restore weights")
    args = ap.parse_args()

    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Restoring archives into project root: {target}")
    if not (target / "scripts").is_dir():
        print("  WARNING: this does not look like the project root (no scripts/ folder found).\n"
              "  Pipeline scripts resolve data/weights relative to the project root and must be run\n"
              "  from there. Re-run with --target <project root> if this location is wrong.")

    if not args.skip_data:
        handle("HandE-Liver3C_data", DATA_URL, DATA_SHA256, args.data_zip, target, target)
    if not args.skip_weights:
        handle("HandE-Liver3C_weights", WEIGHTS_URL, WEIGHTS_SHA256, args.weights_zip, target, target)

    print("\nAll requested archives restored.")


if __name__ == "__main__":
    main()
