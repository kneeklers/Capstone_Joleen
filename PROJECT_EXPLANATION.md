# Project Explanation — What You Need to Know (for Professor)

Use this as your talking points when explaining the capstone to your professor.

---

## 1. What the project does (one sentence)

**Real-time steel-surface defect detection on a Raspberry Pi 5 using a camera:** the system runs a small neural network on the Pi, draws bounding boxes and labels on the live stream, and logs **where** (which zone) and **what** (defect type + confidence) was detected.

---

## 2. Problem & motivation

- **Domain:** Manufacturing / quality control — detecting defects on metal (steel) surfaces.
- **Why on-device (Pi):** Run inference locally without sending video to the cloud; suitable for factory or lab setups.
- **Why object detection (not just classification):** We need **where** the defect is (bounding box) and can have **multiple defects** per image. So we use an object-detection model (YOLO), not a simple image classifier.

---

## 3. The 6 defect classes

The model detects these **6 NEU-DET** defect types:

| Class               | Description (typical)        |
|---------------------|-----------------------------|
| crazing             | Fine crack pattern          |
| inclusion           | Foreign material            |
| patches             | Surface patch               |
| pitted_surface      | Pits / holes                |
| rolled-in_scale     | Scale from rolling process  |
| scratches           | Scratch marks               |

Labels are in `labels.txt`; the model outputs class index + confidence.

---

## 4. Dataset & training (off the Pi)

- **Dataset:** **NEU-DET** (steel surface defect dataset with bounding boxes).
- **Format:** Pascal VOC–style XML annotations → converted to **YOLO format** (one `.txt` per image with `class x_center y_center width height` normalized).
- **Where we train:** **Google Colab** (GPU). We do **not** train on the Pi.
- **Training pipeline (high level):**
  1. Upload `archive-2.zip` (NEU-DET) to Colab.
  2. Convert NEU-DET to YOLO (train/val split, `data.yaml`).
  3. Train **YOLOv8-nano** (small, fast model).
  4. Export to **TFLite** (`best_float32.tflite`) so it can run on the Pi.
- **Outputs we use on the Pi:** `best_float32.tflite` (model), `labels.txt` (class names).

---

## 5. Model & inference (on the Pi)

- **Model:** **YOLOv8-nano**, exported to **TFLite** (TensorFlow Lite).
- **Why TFLite:** Runs efficiently on CPU on the Pi; no need for a GPU.
- **Inference module (`inference.py`):**
  - Loads the TFLite model and labels.
  - Takes a BGR frame (e.g. 640×480), resizes to model input (640×640), runs the interpreter.
  - YOLOv8 output is (8400, 4+6): 8400 proposals, 4 box coordinates + 6 class scores. We decode boxes, apply confidence threshold (0.5) and NMS (IoU 0.45), then map class indices to names (crazing, inclusion, etc.).
  - Returns list of detections: `(x1, y1, x2, y2, class_name, confidence)` in frame coordinates.
- **Drawing:** OpenCV draws rectangles and labels (class + confidence) on the frame.

---

## 6. Raspberry Pi web app (Flask)

- **Role:** Serve the **live camera stream** and run **detection on selected frames**, then show results in the browser.
- **Stack:** **Flask**, **OpenCV**, **TensorFlow** (for TFLite), **NumPy**. No cloud; everything runs on the Pi.

### Camera

- Tries **OpenCV** (V4L2) first. On Pi 5 with Camera Module 3, OpenCV sometimes can’t read frames, so we support:
  - **rpicam-vid** (`USE_RPICAM=1`): same stack as `rpicam-hello`; we parse MJPEG from the process stdout (find JPEG boundaries 0xFFD8 / 0xFFD9 and decode).
  - **picamera2** (optional): if the system has `python3-picamera2` and libcamera.

### Start/Stop analysis

- **Start:** Enables detection, drawing, 3×3 zone tracking, and logging (console + webpage).
- **Stop:** Plain livestream only; no inference, no logging; zone counts are **reset** for the next session.
- Implemented with a global flag (`_analysis_enabled`) and locks so the generator and API stay consistent.

### 3×3 zone tracking (location)

- The **frame is divided into a 3×3 grid** (top_left, top_middle, top_right, middle_left, centre, middle_right, bottom_left, bottom_middle, bottom_right).
- For each detection we take the **center of the bounding box** and assign it to one of these 9 zones.
- We **count detections per zone** and **reset counts when the user stops analysis**.
- The stream shows the grid lines when analysis is on; the webpage shows a 3×3 “zone map” with counts so you can see **where** defects appear (e.g. more in centre or in a corner).

### Web UI

- **Livestream** (left): MJPEG feed; when analysis is on, grid + bounding boxes + labels are drawn.
- **Detection log** (right): **Table** with columns: #, Zone, Defect, Conf, Bbox. Filled from in-memory log lines (format: `[zone] name conf @ (x1,y1)-(x2,y2)`).
- **Zone map:** 3×3 grid of cells with defect count per zone; cells with count &gt; 0 are highlighted.
- **APIs:** `GET /api/analysis?enabled=1|0` (start/stop), `GET /api/logs` (log lines), `GET /api/zones` (counts per zone).

---

## 7. Technical points worth mentioning

1. **Training vs deployment split:** Train on Colab (GPU, PyTorch/Ultralytics), deploy on Pi (TFLite, CPU). No training on the Pi.
2. **Object detection:** Multiple defects per image, with bounding boxes and class labels (not just “defect vs good”).
3. **On-device inference:** All inference and streaming on the Pi; no video sent to the cloud.
4. **Location tracking:** 3×3 zones give a simple “where on the surface” view (e.g. for process or positioning analysis).
5. **User control:** Start/Stop lets the user choose when to run (and log) analysis vs when to just view the stream.
6. **Camera compatibility:** Fallback from OpenCV to rpicam-vid (or picamera2) so it works with Pi Camera Module 3 when V4L2 is problematic.
7. **Thread safety:** Locks used for analysis flag, log buffer, and zone counts so the stream generator and API handlers don’t race.

---

## 8. How to run (recap for demo)

**On the Pi (after boot):**

```bash
cd /path/to/Capstone_Joleen
source venv/bin/activate
USE_RPICAM=1 python app.py   # if using Pi Camera Module 3
```

Then open **http://&lt;pi-ip&gt;:5000** in a browser. Click **Start analysis** to run detection and see the log + zone map; click **Stop** for livestream only.

---

## 9. File roles (quick reference)

| File / folder        | Purpose |
|----------------------|--------|
| `train_on_colab.ipynb` | Colab notebook: convert NEU-DET → YOLO, train YOLOv8-nano, export TFLite |
| `convert_neu_det_to_yolo.py` | NEU-DET (VOC-style) → YOLO format |
| `train_yolo.py`      | Train YOLOv8-nano, export TFLite |
| `best_float32.tflite`| TFLite model (deployed on Pi) |
| `labels.txt`         | Six class names for inference |
| `inference.py`       | Load TFLite, run detection, return boxes + names + confidence; draw helpers |
| `app.py`             | Flask app: camera, stream, 3×3 zones, start/stop, APIs, log |
| `templates/index.html` | Web UI: stream, log table, zone grid, Start/Stop |

Use this document as your script when walking the professor through the problem, data, model, training, deployment, and UI/features.
