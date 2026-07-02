#!/usr/bin/env python3
"""
Package the trained model weights for the three_types release as a single
hosted archive.

By default this ships EVERY fold of every model (``--all-folds``, the setting
used for the published release) so the documented ``--all_folds`` ensemble
inference reproduces the paper exactly. ``--checkpoints best`` keeps only each
fold's ``checkpoint_best.pth`` (what inference loads); ``final`` or ``both`` add
the last-epoch checkpoint.

What it includes
----------------
* nnU-Net (segmentation): for every ``Dataset1NN_LiverX`` it ships the trainer
  directory's inference files (``dataset.json``, ``plans.json``,
  ``dataset_fingerprint.json``) plus the selected checkpoint(s) for the chosen
  folds. ``nnUNet_results/best_folds.json`` (the per-dataset best fold) is also
  included for reference.
* Classification: for every (dataset, model, fold) it ships the selected
  checkpoint(s) plus that fold's ``confusion_matrix_best.json``. A best-fold
  ranking per (dataset, model) is written to ``classification_best_folds.json``
  for convenience (single-fold use).

Archive paths are preserved relative to the project root so
``download_and_extract.py`` restores them into ``nnUNet_results/`` and
``classification_results/`` where the pipeline scripts expect them.

Usage (published release: all folds, best checkpoint each):
    python release_tools/package_weights.py \
        --source /path/to/three_types \
        --output /path/to/output_dir \
        --best-folds-out metadata/classification_best_folds.json \
        --all-folds --checkpoints best

Pass no ``--all-folds`` to ship only each model's single best fold (much smaller).
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

NNUNET_TRAINER_FILES = ("dataset.json", "plans.json", "dataset_fingerprint.json")
CKPT_MAP = {
    "best": ["checkpoint_best.pth"],
    "final": ["checkpoint_final.pth"],
    "both": ["checkpoint_best.pth", "checkpoint_final.pth"],
}


def macro_f1_and_acc(cm_path: Path):
    """Return (macro_f1, accuracy) from a confusion_matrix_best.json, or None."""
    try:
        with open(cm_path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    pcm = d.get("per_class_metrics", {})
    f1s = [v.get("f1") for v in pcm.values() if isinstance(v.get("f1"), (int, float))]
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    matrix = d.get("matrix")
    acc = 0.0
    if matrix:
        total = sum(sum(row) for row in matrix)
        diag = sum(matrix[i][i] for i in range(len(matrix)))
        acc = diag / total if total else 0.0
    return macro_f1, acc


def select_classification_best(source: Path):
    """Build {dataset: {model: {best_fold, macro_f1, accuracy, all_folds:[...]}}}."""
    root = source / "classification_results"
    result = {}
    if not root.exists():
        return result
    for ds in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Dataset")):
        result[ds.name] = {}
        for model in sorted(p for p in ds.iterdir() if p.is_dir()):
            folds = []
            for fold in sorted(p for p in model.iterdir() if p.is_dir() and p.name.startswith("fold_")):
                m = macro_f1_and_acc(fold / "confusion_matrix_best.json")
                if m is None:
                    continue
                folds.append({"fold": int(fold.name.split("_")[1]),
                              "macro_f1": m[0], "accuracy": m[1]})
            if not folds:
                continue
            folds.sort(key=lambda x: (x["macro_f1"], x["accuracy"]), reverse=True)
            best = folds[0]
            result[ds.name][model.name] = {
                "best_fold": best["fold"],
                "macro_f1": best["macro_f1"],
                "accuracy": best["accuracy"],
                "all_folds": folds,
            }
    return result


def collect_nnunet(source: Path, all_folds: bool, ckpts):
    """Yield (abs_path, arcname) for nnU-Net checkpoints + trainer files."""
    results = source / "nnUNet_results"
    if not results.exists():
        print("  WARN: nnUNet_results/ not found; skipping segmentation weights", file=sys.stderr)
        return
    bf_path = results / "best_folds.json"
    best = {}
    if bf_path.is_file():
        yield bf_path, "nnUNet_results/best_folds.json"
        with open(bf_path) as f:
            best = json.load(f)
    for ds_dir in sorted(d for d in results.iterdir() if d.is_dir() and d.name.startswith("Dataset")):
        for cfg_dir in sorted(c for c in ds_dir.iterdir() if c.is_dir()):
            for fn in NNUNET_TRAINER_FILES:
                p = cfg_dir / fn
                if p.is_file():
                    yield p, str(p.relative_to(source)).replace(os.sep, "/")
            if all_folds:
                fold_dirs = sorted(f for f in cfg_dir.iterdir()
                                   if f.is_dir() and f.name.startswith("fold_"))
            else:
                bf = best.get(ds_dir.name, {}).get(cfg_dir.name, {}).get("best_fold")
                fold_dirs = [cfg_dir / f"fold_{bf}"] if bf is not None else []
            for fold_dir in fold_dirs:
                for fn in ckpts:
                    p = fold_dir / fn
                    if p.is_file():
                        yield p, str(p.relative_to(source)).replace(os.sep, "/")


def collect_classification(source: Path, selection: dict, all_folds: bool, ckpts):
    """Yield (abs_path, arcname) for classification checkpoints."""
    root = source / "classification_results"
    for ds_name, models in selection.items():
        for model_name, info in models.items():
            model_dir = root / ds_name / model_name
            if all_folds:
                fold_dirs = sorted(p for p in model_dir.iterdir()
                                   if p.is_dir() and p.name.startswith("fold_"))
            else:
                fold_dirs = [model_dir / f"fold_{info['best_fold']}"]
            for fold_dir in fold_dirs:
                for fn in list(ckpts) + ["confusion_matrix_best.json"]:
                    p = fold_dir / fn
                    if p.is_file():
                        yield p, str(p.relative_to(source)).replace(os.sep, "/")


def write_zip(out_zip: Path, items, compress: bool) -> int:
    # .pth weights are essentially incompressible, so store (no deflate) by
    # default: far faster with negligible size difference.
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    count = 0
    with zipfile.ZipFile(out_zip, "w", method, compresslevel=4, allowZip64=True) as zf:
        for src, arcname in items:
            try:
                zf.write(src, arcname=arcname)
            except ValueError:
                zi = zipfile.ZipInfo(filename=arcname, date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type = method
                with open(src, "rb") as fp:
                    zf.writestr(zi, fp.read())
            count += 1
            if count % 25 == 0:
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
                    help="Full project checkout holding nnUNet_results/ and classification_results/")
    ap.add_argument("--output", type=Path, default=Path.cwd(),
                    help="Directory to write HandE-Liver3C_weights.zip into (default: cwd)")
    ap.add_argument("--best-folds-out", type=Path, default=Path("classification_best_folds.json"),
                    help="Where to also write the classification best-fold ranking JSON")
    ap.add_argument("--all-folds", action="store_true",
                    help="Include every fold of every model (full ensemble reproduction)")
    ap.add_argument("--checkpoints", choices=list(CKPT_MAP), default="best",
                    help="Which checkpoint(s) per fold: best (default), final, or both")
    ap.add_argument("--compress", action="store_true",
                    help="Deflate-compress the archive (slow; .pth barely compresses)")
    args = ap.parse_args()

    source = args.source.resolve()
    ckpts = CKPT_MAP[args.checkpoints]
    args.output.mkdir(parents=True, exist_ok=True)
    out_zip = (args.output / "HandE-Liver3C_weights.zip").resolve()
    if out_zip.exists():
        out_zip.unlink()

    print(f"Source: {source}")
    print(f"Mode:   all_folds={args.all_folds}  checkpoints={args.checkpoints}  compress={args.compress}")
    print("Selecting classification best folds (for reference metadata)...")
    selection = select_classification_best(source)
    args.best_folds_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.best_folds_out, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"  wrote {args.best_folds_out}")

    print(f"\nWriting {out_zip} ...")
    items = list(collect_nnunet(source, args.all_folds, ckpts)) + list(
        collect_classification(source, selection, args.all_folds, ckpts))
    n = write_zip(out_zip, items, args.compress)

    size_gb = out_zip.stat().st_size / (1024 ** 3)
    digest = sha256(out_zip)
    print("\nDone.")
    print(f"  Files:  {n}")
    print(f"  Size:   {size_gb:.2f} GB")
    print(f"  SHA256: {digest}")
    print("\n--> Paste this SHA256 into release_tools/download_and_extract.py (WEIGHTS_SHA256)")


if __name__ == "__main__":
    main()
