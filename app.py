"""
Live defect detection web app for Raspberry Pi 5.
Streams camera and runs TFLite YOLOv8; draws bounding boxes on the livestream.

Run: python app.py
Then open http://localhost:5000 (or http://<pi-ip>:5000 from another device)
Place best_float32.tflite and labels.txt in this directory (or set MODEL_PATH / LABELS_PATH).
"""

import os
import platform
import time
from pathlib import Path
from threading import Lock

import cv2
from flask import Flask, Response, render_template

from inference import DefectDetector, draw_detections

app = Flask(__name__)

# Camera: on Pi (aarch64) we prefer picamera2 for Pi Camera Module 3 / CSI. Set USE_PICAMERA2=0 to use OpenCV instead.
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_SIZE = (640, 480)
_force_picam2 = os.environ.get("USE_PICAMERA2", "").strip().lower() in ("1", "true", "yes")
_force_opencv = os.environ.get("USE_PICAMERA2", "").strip().lower() in ("0", "false", "no")
USE_PICAMERA2 = _force_picam2 or (platform.machine() == "aarch64" and not _force_opencv)
# Run detection every N frames to balance FPS (1 = every frame, 2 = every other, etc.)
DETECT_EVERY_N_FRAME = 1

_camera_lock = Lock()
_camera = None
_picam2 = None
_detector = None


def _try_opencv_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Camera] OpenCV: could not open index {CAMERA_INDEX}. Try CAMERA_INDEX=0 or 1.")
        return None
    if FRAME_SIZE:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    ok, _ = cap.read()
    if not ok:
        print("[Camera] OpenCV: opened but failed to read a frame.")
        cap.release()
        return None
    print("[Camera] OpenCV: using camera index", CAMERA_INDEX)
    return cap


def _try_picamera2():
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        err = str(e)
        if "libcamera" in err:
            print("[Camera] Picamera2 needs the libcamera Python bindings. Run:")
            print("           sudo apt install -y python3-picamera2 python3-libcamera")
            print("         Then recreate venv with system Python:  python3 -m venv venv --system-site-packages")
            print("           source venv/bin/activate && pip install -r requirements.txt")
        elif "picamera2" in err:
            print("[Camera] Picamera2 not visible in this venv. Apt's python3-picamera2 is for system Python.")
            print("         Recreate venv with system Python (so it sees apt packages):")
            print("           deactivate && rm -rf venv")
            print("           python3 -m venv venv --system-site-packages")
            print("           source venv/bin/activate && pip install -r requirements.txt")
            print("         If you used python3.11, apt's picamera2 may be for a different version; use python3 instead.")
        else:
            print("[Camera] Picamera2: not installed.", e)
        return None
    try:
        picam2 = Picamera2()
        # Pi Camera Module 3 (imx708) supports 1536x864, 2304x1296, 4608x2592. Use 1536x864 then we resize to FRAME_SIZE.
        for size in [FRAME_SIZE, (1536, 864), (2304, 1296)]:
            try:
                config = picam2.create_preview_configuration(
                    main={"size": size, "format": "RGB888"}
                )
                picam2.configure(config)
                break
            except Exception:
                continue
        else:
            config = picam2.create_preview_configuration(main={"format": "RGB888"})
            picam2.configure(config)
        picam2.start()
        time.sleep(1)
        print("[Camera] Picamera2: Pi Camera Module (imx708) is in use.")
        return picam2
    except Exception as e:
        print("[Camera] Picamera2: failed to start. Enable camera: sudo raspi-config -> Interface Options -> Camera -> Enable")
        print("         Error:", e)
        return None


def get_camera():
    """Return OpenCV VideoCapture or Picamera2. On Pi (aarch64) prefer Picamera2 for Camera Module 3."""
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
        print("[Camera] No camera available. See messages above. Try: USE_PICAMERA2=1 python app.py  or  USE_PICAMERA2=0 python app.py")
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
                if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                    frame = cv2.resize(frame, FRAME_SIZE)
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
