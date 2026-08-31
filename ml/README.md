# ML component

Detection + segmentation + tracking + depth + severity. No GPS, no DB, no heatmap here —
this module outputs structured events that `backend/` consumes.

## Setup

```bash
cd ml
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Folders

- `data/` — datasets (gitignored — download instructions below, not committed to git)
- `weights/` — trained/downloaded model weights (gitignored)
- `configs/` — YOLO dataset config files (`.yaml`)
- `src/` — actual code (`train.py`, `pipeline.py`, `severity.py`)
- `notebooks/` — exploration/debugging notebooks

## Datasets (download separately, do not commit)

- UDTIRI: https://www.kaggle.com/datasets/jiahangli617/udtiri
- Roboflow pothole segmentation: https://universe.roboflow.com/pothole-vsmtu/potholes-and-roads-instance-segmentation

## Status

- [ ] Train YOLOv8n-seg on merged pothole dataset
- [ ] Wire up ByteTrack for duplicate suppression
- [ ] Add Depth Anything V2-Small for relative depth
- [ ] Write severity scoring function
- [ ] Export to ONNX for edge deployment
