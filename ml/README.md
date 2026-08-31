# Machine Learning Pipeline (`ml/`)

This directory contains the core Computer Vision and Machine Learning components for the Edge Vision Road Anomaly project. 

Our pipeline is responsible for **detection, instance segmentation, object tracking, relative depth estimation, and severity scoring** of road anomalies like potholes. 

> [!NOTE]
> This module operates independently of GPS, database layers, and heatmap generation. Its sole purpose is to process video/image frames and output structured anomaly events to be consumed by the `backend/`.

## 📂 Directory Structure

- `configs/` — Configuration files (e.g., YOLO dataset YAMLs).
- `data/` — Local storage for datasets. **(Ignored in git. Download manually!)**
- `notebooks/` — Jupyter notebooks for data exploration, testing, and debugging.
- `src/` — The source code for the ML pipeline:
  - `fix_labels.py` — Utility script to clean and format Roboflow dataset labels for YOLOv8.
  - `train.py` — Main training script to fine-tune YOLOv8-seg models on our custom pothole dataset. Includes custom F1 and mAP metric logging.
- `weights/` — Trained and downloaded model weights (e.g., `best.pt`, `yolov8n-seg.pt`). **(Ignored in git)**

## 🚀 Setup Instructions

1. **Navigate to the ML directory:**
   ```bash
   cd ml
   ```

2. **Create a Python Virtual Environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the Environment:**
   - **Windows (Command Prompt):** `.venv\Scripts\activate.bat`
   - **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
   - **Mac/Linux:** `source .venv/bin/activate`

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > [!TIP]
   > If you have an NVIDIA GPU, you may want to install PyTorch with CUDA support to drastically speed up training. You can do this by running `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` inside your virtual environment.

## 💾 Datasets

We use the following datasets for training. **Do not commit these to git.** Download them and place them in the `data/` folder.

1. **Roboflow Pothole Segmentation** (Main dataset):
   - Link: [Roboflow Universe](https://universe.roboflow.com/pothole-vsmtu/potholes-and-roads-instance-segmentation)
2. **UDTIRI Dataset** (Supplemental):
   - Link: [Kaggle](https://www.kaggle.com/datasets/jiahangli617/udtiri)

## 🏃‍♂️ Running the Code

### 1. Fix Dataset Labels
Before training, ensure your Roboflow labels are correctly formatted for YOLOv8 segmentation:
```bash
python src/fix_labels.py
```

### 2. Train the Model
Fine-tune the YOLOv8n-seg model on the pothole dataset. By default, it runs for 100 epochs and logs metrics.
```bash
python src/train.py
```

## 📈 Current Project Status & Roadmap

- [x] Train YOLOv8n-seg on merged pothole dataset
- [ ] Wire up **ByteTrack** for duplicate suppression (tracking identical potholes across frames).
- [ ] Integrate **Depth Anything V2-Small** for relative depth estimation (measuring how deep a pothole is).
- [ ] Write the **severity scoring function** (combining depth, area, and class to rate danger).
- [ ] Export the final pipeline to **ONNX** for optimized edge deployment.
