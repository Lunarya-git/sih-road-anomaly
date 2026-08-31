"""
Fine-tune YOLOv8n-seg (COCO-pretrained) on the pothole segmentation dataset.

Metrics logged per epoch
------------------------
- Box  : Precision, Recall, F1, mAP50, mAP50-95
- Mask : Precision, Recall, F1, mAP50, mAP50-95

Run fix_labels.py first to clean the dataset labels.

Usage:
    python src/train.py
"""

import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Resolve everything relative to this file so the script works from any CWD.
_ML_ROOT = Path(__file__).resolve().parent.parent  # .../sih-road-anomaly/ml

DATA_CONFIG = _ML_ROOT / "configs" / "pothole-seg.yaml"
BASE_WEIGHTS = "yolov8n-seg.pt"   # auto-downloads on first run
EPOCHS = 100
IMG_SIZE = 640
PROJECT_DIR = str(_ML_ROOT / "weights" / "runs")
RUN_NAME = "pothole_seg_v1"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; returns 0 when both are 0."""
    return 2 * precision * recall / (precision + recall + 1e-9)


def _make_epoch_callback(epoch_log: list) -> callable:
    """Return an on_fit_epoch_end callback that prints and records F1."""

    def on_fit_epoch_end(trainer) -> None:
        m = trainer.metrics          # dict populated by Ultralytics after val
        epoch = trainer.epoch + 1    # 1-based for display

        # --- box metrics ---
        bp = m.get("metrics/precision(B)", 0.0)
        br = m.get("metrics/recall(B)", 0.0)
        b_map50 = m.get("metrics/mAP50(B)", 0.0)
        b_map5095 = m.get("metrics/mAP50-95(B)", 0.0)
        bf1 = _f1(bp, br)

        # --- mask metrics ---
        mp = m.get("metrics/precision(M)", 0.0)
        mr = m.get("metrics/recall(M)", 0.0)
        m_map50 = m.get("metrics/mAP50(M)", 0.0)
        m_map5095 = m.get("metrics/mAP50-95(M)", 0.0)
        mf1 = _f1(mp, mr)

        row = {
            "epoch": epoch,
            "box/P": bp, "box/R": br, "box/F1": bf1,
            "box/mAP50": b_map50, "box/mAP50-95": b_map5095,
            "mask/P": mp, "mask/R": mr, "mask/F1": mf1,
            "mask/mAP50": m_map50, "mask/mAP50-95": m_map5095,
        }
        epoch_log.append(row)

        print(
            f"\n  [Metrics] Epoch {epoch:3d} | "
            f"Box  P={bp:.3f} R={br:.3f} F1={bf1:.3f} mAP50={b_map50:.3f} | "
            f"Mask P={mp:.3f} R={mr:.3f} F1={mf1:.3f} mAP50={m_map50:.3f}\n"
        )

    return on_fit_epoch_end


def print_summary(epoch_log: list, save_dir: str) -> None:
    """Print a compact training summary and save metrics to CSV."""
    if not epoch_log:
        return

    # Find best epoch by mask mAP50
    best = max(epoch_log, key=lambda r: r["mask/mAP50"])

    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    header = (
        f"  {'Epoch':>5}  {'Box-F1':>7}  {'Box-mAP50':>10}  "
        f"{'Mask-F1':>8}  {'Mask-mAP50':>11}  {'Mask-mAP50-95':>14}"
    )
    print(header)
    print("  " + "-" * 66)
    for r in epoch_log:
        marker = " ◀ best" if r is best else ""
        print(
            f"  {r['epoch']:5d}  {r['box/F1']:7.4f}  {r['box/mAP50']:10.4f}  "
            f"{r['mask/F1']:8.4f}  {r['mask/mAP50']:11.4f}  "
            f"{r['mask/mAP50-95']:14.4f}{marker}"
        )
    print("=" * 70)
    print(f"  Best epoch : {best['epoch']}  (Mask mAP50={best['mask/mAP50']:.4f})")
    print(f"  Results dir: {save_dir}")
    print("=" * 70 + "\n")

    # Save a quick "best results" text file for easy reading
    txt_path = Path(save_dir) / "best_results.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("=================================================\n")
        fh.write(f" BEST RESULTS SUMMARY (Run: {Path(save_dir).name})\n")
        fh.write("=================================================\n\n")
        fh.write(f"Best Epoch     : {best['epoch']}\n\n")
        fh.write("--- MASK (Segmentation) Metrics ---\n")
        fh.write(f"mAP50          : {best['mask/mAP50']:.4f}\n")
        fh.write(f"mAP50-95       : {best['mask/mAP50-95']:.4f}\n")
        fh.write(f"F1 Score       : {best['mask/F1']:.4f}\n")
        fh.write(f"Precision      : {best['mask/P']:.4f}\n")
        fh.write(f"Recall         : {best['mask/R']:.4f}\n\n")
        fh.write("--- BOX (Detection) Metrics ---\n")
        fh.write(f"mAP50          : {best['box/mAP50']:.4f}\n")
        fh.write(f"mAP50-95       : {best['box/mAP50-95']:.4f}\n")
        fh.write(f"F1 Score       : {best['box/F1']:.4f}\n")
        fh.write(f"Precision      : {best['box/P']:.4f}\n")
        fh.write(f"Recall         : {best['box/R']:.4f}\n\n")
        fh.write("=================================================\n")
        fh.write(f"Best weights   : weights/best.pt\n")
        fh.write("=================================================\n")

    # Save CSV alongside the Ultralytics results.csv
    csv_path = Path(save_dir) / "f1_metrics.csv"
    import csv
    fieldnames = list(epoch_log[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_log)
    print(f"  Best results summary saved to: {txt_path}")
    print(f"  F1 metrics saved to: {csv_path}\n")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    model = YOLO(BASE_WEIGHTS)

    # Ultralytics resolves relative `path` values in data YAMLs against its
    # own DATASETS_DIR setting (not the YAML file location).  To avoid that,
    # we load the YAML, inject an absolute dataset path, and write a temp file
    # that Ultralytics reads directly.
    with open(DATA_CONFIG, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    data_cfg["path"] = str(_ML_ROOT / "data" / "potholes_roboflow")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(data_cfg, tmp)
        tmp_path = tmp.name

    # Accumulate per-epoch metrics via callback
    epoch_log: list = []
    model.add_callback("on_fit_epoch_end", _make_epoch_callback(epoch_log))

    try:
        results = model.train(
            data=tmp_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            project=PROJECT_DIR,
            name=RUN_NAME,
            patience=20,       # early-stop if val metrics stall for 20 epochs
            plots=True,
            device=0
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Print summary table and save F1 CSV
    save_dir = str(results.save_dir) if results else f"{PROJECT_DIR}/{RUN_NAME}"
    print_summary(epoch_log, save_dir)

    print("Best weights saved at:")
    print(f"  {save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
