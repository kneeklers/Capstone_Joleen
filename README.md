# Capstone_Joleen

Defect detection on Raspberry Pi 5 with camera. Train on Colab/Kaggle, export to TFLite, run on the Pi. Supports **6 defect types** (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches) with **multiple defects per image** via YOLOv8-nano.

## Train on Google Colab (one notebook)

**Open and run this in Colab:** [`train_on_colab.ipynb`](train_on_colab.ipynb)

1. In Colab: **File → Upload notebook** and choose `train_on_colab.ipynb` (or clone this repo and open the notebook).
2. Run all cells in order. When prompted, **upload `archive-2.zip`** (your NEU-DET dataset).
3. The notebook will install deps, clone this repo for the scripts, convert NEU-DET to YOLO, train YOLOv8-nano, export TFLite, and offer a zip to download for the Pi.

No need to run `convert_neu_det_to_yolo.py` or `train_yolo.py` by hand—the notebook runs them for you.

---

## Training from CLI (optional)

If you prefer to run scripts yourself (e.g. on Kaggle or locally):

For NEU-DET–style data (bounding boxes, multiple defects per image), use object detection.

### Step 1: Unzip and convert NEU-DET to YOLO format

After uploading `archive-2.zip` (or the Kaggle dataset) to your runtime, unzip and convert:

```bash
unzip -q archive-2.zip -d /content
pip install -r requirements-train.txt

python convert_neu_det_to_yolo.py \
  --input-dir /content/NEU-DET \
  --output-dir /content/yolo_dataset \
  --val-ratio 0.2
```

This creates `yolo_dataset/train/`, `yolo_dataset/val/`, and `yolo_dataset/data.yaml`.

### Step 2: Train YOLOv8-nano and export TFLite

```bash
python train_yolo.py \
  --data-dir /content/yolo_dataset \
  --output-dir /content/output \
  --epochs 80 \
  --imgsz 640
```

Use `--imgsz 320` for a lighter model on the Pi if needed.

### Outputs (YOLO path)

- `output/yolov8n_defect/weights/best.pt` – best PyTorch weights
- `output/*.tflite` – TFLite model for the Pi
- `output/labels.txt` – defect class names (for inference)

---

## Alternative: classification only (defect vs good)

If you only need “defect vs good” per image (no bounding boxes), use the MobileNetV2 classifier:

- Dataset: folder layout `train/defect/`, `train/good/`, `val/defect/`, `val/good/`.
- Run: `python train.py --data-dir <DATA_DIR> --output-dir <OUTPUT_DIR> --epochs 15`
- Outputs: `defect_mobilenetv2.tflite`, `labels.txt`.

---

## Run the web app on the Pi 5 (live camera)

A **Flask** app streams the camera and runs **live defect detection** with bounding boxes on the stream.

1. Copy your Colab outputs to the project folder (same directory as `app.py`):
   - **best_float32.tflite** (or best_float16.tflite)
   - **labels.txt**

2. On the Pi, install and run:

```bash
pip install -r requirements.txt
python app.py
```

3. Open in a browser:
   - On the Pi: **http://localhost:5000**
   - From another device: **http://\<pi-ip\>:5000**

You’ll see the live feed with defect boxes and labels. If the model/labels are missing, the app still streams video without detections.

---

## Colab: "No .tflite in /content/output"

Ultralytics sometimes saves the TFLite file under `yolov8n_defect/weights/best_saved_model/` instead of `/content/output`. Two options:

1. **Re-run the download cell** — The notebook’s last cell now looks in that folder. If your notebook is old, copy the latest `train_on_colab.ipynb` from this repo and run the last cell again.

2. **Or run this in a new Colab cell** (after training) to find and download the zip:

```python
from pathlib import Path
from google.colab import files

output_dir = Path('/content/output')
weights_dir = output_dir / 'yolov8n_defect' / 'weights'
tflite = list(output_dir.glob('*.tflite')) or list(weights_dir.glob('*.tflite')) + list((weights_dir / 'best_saved_model').glob('*.tflite'))
tflite = [str(p) for p in tflite]
labels_path = output_dir / 'labels.txt'

if tflite and labels_path.exists():
    import subprocess
    subprocess.run(['zip', '-j', 'defect_model_for_pi.zip'] + tflite + [str(labels_path)], check=True)
    files.download('defect_model_for_pi.zip')
else:
    print('TFLite:', tflite)
    print('Labels:', labels_path.exists())
```