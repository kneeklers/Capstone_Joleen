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

# Camera: 0 = default (USB or Pi camera via v4l2). Set USE_PICAMERA2=1 to force Pi Camera Module (picamera2).
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_SIZE = (640, 480)
USE_PICAMERA2 = os.environ.get("USE_PICAMERA2", "").strip().lower() in ("1", "true", "yes")
# Run detection every N frames to balance FPS (1 = every frame, 2 = every other, etc.)
DETECT_EVERY_N_FRAME = 1

_camera_lock = Lock()
_camera = None
_picam2 = None
_detector = None


def _try_opencv_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return None
    if FRAME_SIZE:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    # Probe one frame
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def _try_picamera2():
    try:
        from picamera2 import Picamera2
    except ImportError:
        return None
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": FRAME_SIZE, "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)  # let camera settle
        return picam2
    except Exception:
        return None


def get_camera():
    """Return OpenCV VideoCapture or Picamera2. Prefer OpenCV unless USE_PICAMERA2=1 or OpenCV fails."""
    global _camera, _picam2
    with _camera_lock:
        if _camera is not None:
            return ("opencv", _camera)
        if _picam2 is not None:
            return ("picam2", _picam2)
        if USE_PICAMERA2:
            _picam2 = _try_picamera2()
            if _picam2 is not None:
                return ("picam2", _picam2)
            _camera = _try_opencv_camera()
        else:
            _camera = _try_opencv_camera()
            if _camera is None:
                _picam2 = _try_picamera2()
                if _picam2 is not None:
                    return ("picam2", _picam2)
        if _camera is not None:
            return ("opencv", _camera)
        return (None, None)


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


def _no_camera_frame():
    """Return a single BGR frame with 'No camera' message."""
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    cv2.putText(
        frame, "No camera found. Check CAMERA_INDEX or USE_PICAMERA2=1 for Pi Camera.",
        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )
    return frame


def generate_frames():
    """Yield JPEG frames for MJPEG live stream, with optional defect detection and bounding boxes."""
    backend, cam = get_camera()
    if backend is None or cam is None:
        frame = _no_camera_frame()
        _, jpeg = cv2.imencode(".jpg", frame)
        while True:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(1)
        return
    detector = get_detector()
    frame_count = 0
    while True:
        if backend == "opencv":
            success, frame = cam.read()
            if not success:
                break
        else:
            try:
                frame = cam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
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
