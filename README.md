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

A **Flask** app serves a live camera stream in the browser. No model is used yet; you can add TFLite inference later.

1. On the Pi, install and run:

```bash
pip install -r requirements.txt
python app.py
```

2. Open in a browser:
   - On the Pi: **http://localhost:5000**
   - From another device on the same network: **http://\<pi-ip\>:5000**

You’ll see the live feed. To add defect detection later, plug your TFLite model and labels into `app.py` (see the `TODO` in `generate_frames()`).