"""
Live defect detection web app for Raspberry Pi 5.
Streams camera and runs TFLite YOLOv8; draws bounding boxes on the livestream.

Run: python app.py
Then open http://localhost:5000 (or http://<pi-ip>:5000 from another device)
Place best_float32.tflite and labels.txt in this directory (or set MODEL_PATH / LABELS_PATH).
"""

import os
import time
from pathlib import Path
from threading import Lock

import cv2
from flask import Flask, Response, render_template

from inference import DefectDetector, draw_detections

app = Flask(__name__)

# Camera: use 0 for default (Pi camera module or USB). Change if needed.
CAMERA_INDEX = 0
FRAME_SIZE = (640, 480)
# Run detection every N frames to balance FPS (1 = every frame, 2 = every other, etc.)
DETECT_EVERY_N_FRAME = 1

_camera_lock = Lock()
_camera = None
_detector = None


def get_camera():
    global _camera
    with _camera_lock:
        if _camera is None:
            _camera = cv2.VideoCapture(CAMERA_INDEX)
            if FRAME_SIZE:
                _camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
                _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        return _camera


def get_detector():
    """Lazy-init TFLite detector. Returns None if model/labels not found."""
    global _detector
    if _detector is not None:
        return _detector
    if DefectDetector is None:
        return None
    base = Path(__file__).resolve().parent
    model = base / "best_float32.tflite"
    if not model.is_file():
        model = base / "best_float16.tflite"
    labels = base / "labels.txt"
    if model.is_file() and labels.is_file():
        try:
            _detector = DefectDetector(model_path=str(model), labels_path=str(labels))
        except Exception:
            _detector = None
    return _detector


def generate_frames():
    """Yield JPEG frames with live defect detection and bounding boxes."""
    cam = get_camera()
    detector = get_detector()
    frame_count = 0
    while True:
        success, frame = cam.read()
        if not success:
            break
        if detector is not None and (frame_count % DETECT_EVERY_N_FRAME == 0):
            detections = detector.detect(frame)
            draw_detections(frame, detections)
        frame_count += 1
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(1 / 15)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    # 0.0.0.0 so you can open from another device on the network
    app.run(host="0.0.0.0", port=5000, threaded=True)
