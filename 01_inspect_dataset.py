"""
01_inspect_dataset.py
----------------------
Run this FIRST, before training.

It auto-detects your dataset layout (flat class folders, or train/ + test/
subfolders each containing class folders), and reports:
  - number of classes
  - class names
  - image count per class
  - corrupted / unreadable images (and quarantines them, doesn't delete)

It writes:
  - dataset_report.json   (machine-readable summary used by train.py)
  - corrupted_files.txt   (list of bad files found)

Usage:
    python 01_inspect_dataset.py --data_dir "D:/path/to/skin-diseases-image-dataset"
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dirs(data_dir: Path):
    """
    Handles two common Kaggle layouts:
      A) data_dir/<class_name>/*.jpg                       (flat)
      B) data_dir/train/<class_name>/*.jpg
         data_dir/test/<class_name>/*.jpg                  (pre-split)
    Returns: layout ("flat" | "split"), dict[class_name] -> list[Path to class dirs]
    """
    entries = [p for p in data_dir.iterdir() if p.is_dir()]
    lower_names = {p.name.lower(): p for p in entries}

    split_keys = [k for k in ("train", "training") if k in lower_names]
    test_keys = [k for k in ("test", "val", "valid", "validation") if k in lower_names]

    if split_keys:
        # layout B: pre-split. We'll merge train(+test) per class, and let
        # train.py redo an 80/20 split itself for consistency, unless you
        # tell it to respect the existing split (see train.py --respect_split).
        class_to_dirs = defaultdict(list)
        for key in split_keys + test_keys:
            split_dir = lower_names[key]
            for cls_dir in split_dir.iterdir():
                if cls_dir.is_dir():
                    class_to_dirs[cls_dir.name].append(cls_dir)
        return "split", class_to_dirs

    # layout A: flat class folders directly under data_dir
    class_to_dirs = {p.name: [p] for p in entries}
    return "flat", class_to_dirs


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # cheap structural check
        # re-open and force full decode (verify() alone can miss truncation)
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to the extracted Kaggle dataset")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    layout, class_to_dirs = find_class_dirs(data_dir)
    print(f"Detected layout: {layout}")
    print(f"Found {len(class_to_dirs)} class folders\n")

    report = {
        "data_dir": str(data_dir),
        "layout": layout,
        "num_classes": len(class_to_dirs),
        "class_names": sorted(class_to_dirs.keys()),
        "per_class_counts": {},
        "per_class_corrupted": {},
        "total_images": 0,
        "total_corrupted": 0,
    }

    corrupted_log = []

    for cls_name in sorted(class_to_dirs.keys()):
        dirs = class_to_dirs[cls_name]
        files = []
        for d in dirs:
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS:
                    files.append(f)

        n_total = len(files)
        n_bad = 0
        for f in files:
            if not is_valid_image(f):
                n_bad += 1
                corrupted_log.append(str(f))

        report["per_class_counts"][cls_name] = n_total
        report["per_class_corrupted"][cls_name] = n_bad
        report["total_images"] += n_total
        report["total_corrupted"] += n_bad

        print(f"  {cls_name:35s}  images={n_total:5d}   corrupted={n_bad}")

    print(f"\nTOTAL images   : {report['total_images']}")
    print(f"TOTAL corrupted: {report['total_corrupted']}")

    out_dir = Path(__file__).parent
    with open(out_dir / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "corrupted_files.txt", "w") as f:
        f.write("\n".join(corrupted_log))

    print(f"\nWrote dataset_report.json and corrupted_files.txt to {out_dir}")
    if report["total_corrupted"] > 0:
        print(
            "\nNOTE: corrupted files were NOT deleted automatically. "
            "Review corrupted_files.txt, then either delete them manually or "
            "re-run with train.py's --skip_corrupted flag (on by default) "
            "which filters them out at load time using this report."
        )
    if report["num_classes"] < 2:
        print("\nWARNING: fewer than 2 class folders detected — check --data_dir path.")


if __name__ == "__main__":
    main()
