"""
Fix mixed box/segment label files exported by Roboflow.

Problem
-------
Some label lines are bounding-box format  (5 fields:  class xc yc w h)
while others are polygon/segmentation format (>5 fields: class x1 y1 x2 y2 …).
Ultralytics discards any label file that mixes both formats, silently shrinking
your effective training set.

Fix
---
* Keep ONLY class 0 (pothole).
* Convert any box-format pothole line (5 fields) into a 4-corner rectangle
  polygon so every line in every file is consistent polygon format.
* Drop class 1 (road) entirely — it is not needed for road-anomaly detection.
* Back up the original file as <name>.txt.bak before modifying.
* Write an empty file (not delete) when a label has no pothole lines so the
  image is treated as background, which is fine for training.

Usage
-----
    python src/fix_labels.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

SPLITS = ["train", "valid", "test"]
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "potholes_roboflow"
POTHOLE_CLASS_ID = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def box_to_polygon(xc: float, yc: float, w: float, h: float) -> list:
    """Convert YOLO box (xc, yc, w, h) to 4-corner clockwise polygon coords."""
    x1, y1 = xc - w / 2, yc - h / 2  # top-left
    x2, y2 = xc + w / 2, yc - h / 2  # top-right
    x3, y3 = xc + w / 2, yc + h / 2  # bottom-right
    x4, y4 = xc - w / 2, yc + h / 2  # bottom-left
    # Clamp to [0, 1] to handle any floating-point edge cases
    coords = [x1, y1, x2, y2, x3, y3, x4, y4]
    return [max(0.0, min(1.0, v)) for v in coords]


def fix_label_file(label_path, dry_run):
    """
    Process a single label file.

    Returns (n_kept, n_converted, n_dropped, changed)
    """
    raw = label_path.read_text(encoding="utf-8").splitlines()

    fixed_lines = []
    n_kept = n_converted = n_dropped = 0

    for line in raw:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        try:
            cls = int(parts[0])
        except (ValueError, IndexError):
            continue  # malformed line — skip silently

        if cls != POTHOLE_CLASS_ID:
            n_dropped += 1
            continue  # drop road (class 1) and anything else

        coords_raw = parts[1:]
        n_coords = len(coords_raw)

        if n_coords == 4:
            # Bounding-box format: xc yc w h → convert to 4-corner polygon
            try:
                xc, yc, w, h = map(float, coords_raw)
            except ValueError:
                continue
            poly = box_to_polygon(xc, yc, w, h)
            fixed_line = "0 " + " ".join(f"{v:.8f}" for v in poly)
            n_converted += 1
        elif n_coords >= 6 and n_coords % 2 == 0:
            # Already polygon format — keep as-is, normalise class to 0
            fixed_line = "0 " + " ".join(coords_raw)
            n_kept += 1
        else:
            # Odd / unexpected field count — skip (don't corrupt the file)
            continue

        fixed_lines.append(fixed_line)

    original_lines = [l.strip() for l in raw if l.strip()]
    changed = fixed_lines != original_lines

    if not dry_run and changed:
        # Back up original (only once)
        bak_path = label_path.with_suffix(".txt.bak")
        if not bak_path.exists():
            shutil.copy2(label_path, bak_path)
        # Write fixed content — empty file is valid (image = background)
        label_path.write_text(
            "\n".join(fixed_lines) + ("\n" if fixed_lines else ""),
            encoding="utf-8",
        )

    return n_kept, n_converted, n_dropped, changed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run=False):
    prefix = "[DRY RUN] " if dry_run else ""

    total_files = 0
    total_changed = 0
    total_kept = 0
    total_converted = 0
    total_dropped = 0
    total_empty = 0

    for split in SPLITS:
        label_dir = DATA_ROOT / split / "labels"
        if not label_dir.exists():
            print(f"  [skip] {label_dir} not found")
            continue

        label_files = sorted(label_dir.glob("*.txt"))
        split_changed = 0

        for lf in label_files:
            n_kept, n_converted, n_dropped, changed = fix_label_file(lf, dry_run)
            total_files += 1
            total_kept += n_kept
            total_converted += n_converted
            total_dropped += n_dropped
            if changed:
                total_changed += 1
                split_changed += 1
            if n_kept + n_converted == 0:
                total_empty += 1

        print(
            f"  {prefix}{split:6s}: {len(label_files)} files, "
            f"{split_changed} modified"
        )

    print()
    print("=" * 55)
    print(f"  Total label files  : {total_files}")
    print(f"  Files modified     : {total_changed}")
    print(f"  Polygon lines kept : {total_kept}")
    print(f"  Box lines converted: {total_converted}  (box -> 4-corner polygon)")
    print(f"  Road lines dropped : {total_dropped}")
    print(f"  Files now empty    : {total_empty}  (no potholes = background)")

    if not dry_run:
        print()
        print("  Originals backed up as <name>.txt.bak")
        print("  Deleting stale Ultralytics cache files...")
        for split in SPLITS:
            cache = DATA_ROOT / split / "labels.cache"
            if cache.exists():
                cache.unlink()
                print(f"    Deleted: {cache}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix mixed box/segment Roboflow labels.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
