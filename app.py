"""
Live defect detection web app for Raspberry Pi 5.
Streams camera and runs TFLite YOLOv8; draws bounding boxes on the livestream.

Run: python app.py
Then open http://localhost:5000 (or http://<pi-ip>:5000 from another device)
Place best_float32.tflite and labels.txt in this directory (or set MODEL_PATH / LABELS_PATH).
"""

# Python 3.12+ removed the "imp" module; TensorFlow's flatbuffers still expects it. Provide a minimal shim.
import sys
if sys.version_info >= (3, 12):
    import importlib.util
    _imp = type(sys)("imp")
    def _find_module(name, path=None):
        try:
            return importlib.util.find_spec(name)
        except ModuleNotFoundError:
            return None
    _imp.find_module = _find_module
    sys.modules["imp"] = _imp

import os
import platform
import subprocess
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, render_template

from inference import DefectDetector, draw_detections

app = Flask(__name__)

# Camera: OpenCV first (simplest). Set USE_RPICAM=1 or USE_PICAMERA2=1 to try other backends.
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_SIZE = (640, 480)
USE_RPICAM = os.environ.get("USE_RPICAM", "").strip().lower() in ("1", "true", "yes")
USE_PICAMERA2 = os.environ.get("USE_PICAMERA2", "").strip().lower() in ("1", "true", "yes")
DETECT_EVERY_N_FRAME = 1

_camera_lock = Lock()
_camera = None
_picam2 = None
_rpicam_proc = None
_detector = None

# Start/Stop analysis: when True, run detection + draw boxes + log; when False, plain livestream only.
_analysis_enabled = False
_analysis_lock = Lock()


def _try_opencv_camera():
    # Pi Camera (rp1-cfe) /dev/video0 supports YUYV, BGR3, RGB3 - no MJPEG. Set format then size.
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            continue
        for fourcc_name, fourcc_val in (
            ("BGR3", cv2.VideoWriter_fourcc(*"BGR3")),   # 24-bit BGR, native OpenCV
            ("YUYV", cv2.VideoWriter_fourcc(*"YUYV")),  # common V4L2
            ("RGB3", cv2.VideoWriter_fourcc(*"RGB3")),
            ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
            (None, None),
        ):
            if fourcc_val is not None:
                cap.set(cv2.CAP_PROP_FOURCC, fourcc_val)
            if FRAME_SIZE:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
            # Some V4L2 drivers need a couple of reads before they deliver
            for _ in range(3):
                ok, _ = cap.read()
                if ok:
                    break
            if ok:
                msg = f"format {fourcc_name}" if fourcc_name else ""
                print("[Camera] OpenCV: using camera index", CAMERA_INDEX, msg)
                return cap
        cap.release()
    print(f"[Camera] OpenCV: could not read from index {CAMERA_INDEX}. Use USE_RPICAM=1 for Pi Camera.")
    return None


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


def _try_rpicam_vid():
    """Use rpicam-vid (same C stack as rpicam-hello). No Python picamera2/libcamera needed."""
    global _rpicam_proc
    for cmd in (
        ["rpicam-vid", "-t", "0", "-n", "--width", str(FRAME_SIZE[0]), "--height", str(FRAME_SIZE[1]), "--codec", "mjpeg", "-o", "-"],
        ["libcamera-vid", "-t", "0", "-n", "--width", str(FRAME_SIZE[0]), "--height", str(FRAME_SIZE[1]), "--codec", "mjpeg", "-o", "-"],
    ):
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            # Give it a moment to start
            time.sleep(1.5)
            if proc.poll() is not None:
                continue
            _rpicam_proc = proc
            print("[Camera] Using rpicam-vid (same as rpicam-hello). No Python camera bindings needed.")
            return proc
        except FileNotFoundError:
            continue
    return None


def _read_rpicam_frame(proc):
    """Read one MJPEG frame from rpicam-vid stdout (JPEG start 0xFFD8, end 0xFFD9)."""
    SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    buf = b""
    while True:
        chunk = proc.stdout.read(65536)
        if not chunk:
            return None
        buf += chunk
        if SOI in buf and EOI in buf:
            start = buf.index(SOI)
            end = buf.index(EOI, start) + 2
            jpeg = buf[start:end]
            buf = buf[end:]
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
    return None


def get_camera():
    """Return (backend, handle). OpenCV is tried first; use USE_RPICAM=1 or USE_PICAMERA2=1 for other backends."""
    global _camera, _picam2, _rpicam_proc
    with _camera_lock:
        if _camera is not None:
            return ("opencv", _camera)
        if _picam2 is not None:
            return ("picam2", _picam2)
        if _rpicam_proc is not None:
            return ("rpicam", _rpicam_proc)
        # Always try OpenCV first (simplest)
        _camera = _try_opencv_camera()
        if _camera is not None:
            return ("opencv", _camera)
        if USE_RPICAM:
            _rpicam_proc = _try_rpicam_vid()
            if _rpicam_proc is not None:
                return ("rpicam", _rpicam_proc)
        if USE_PICAMERA2:
            _picam2 = _try_picamera2()
            if _picam2 is not None:
                return ("picam2", _picam2)
        # Fallback: try rpicam on Pi if OpenCV failed
        if platform.machine() == "aarch64":
            _rpicam_proc = _try_rpicam_vid()
            if _rpicam_proc is not None:
                return ("rpicam", _rpicam_proc)
            _picam2 = _try_picamera2()
            if _picam2 is not None:
                return ("picam2", _picam2)
        print("[Camera] No camera available. Try CAMERA_INDEX=0 or 1, or USE_RPICAM=1 for Pi Camera Module.")
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
    """Yield JPEG frames for MJPEG live stream with defect bounding boxes and labels; print detections to console."""
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
    last_detections = []
    last_print_time = 0.0
    while True:
        if backend == "opencv":
            success, frame = cam.read()
            if not success:
                break
        elif backend == "rpicam":
            frame = _read_rpicam_frame(cam)
            if frame is None:
                break
            if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                frame = cv2.resize(frame, FRAME_SIZE)
        else:
            try:
                frame = cam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                    frame = cv2.resize(frame, FRAME_SIZE)
            except Exception:
                break
        with _analysis_lock:
            do_analysis = _analysis_enabled
        if do_analysis and detector is not None and (frame_count % DETECT_EVERY_N_FRAME == 0):
            last_detections = detector.detect(frame)
            if last_detections and (time.time() - last_print_time) >= 1.0:
                for x1, y1, x2, y2, name, conf in last_detections:
                    print(f"[Defect] {name} {conf:.2f} @ ({x1},{y1})-({x2},{y2})")
                last_print_time = time.time()
        elif not do_analysis:
            last_detections = []
        if last_detections:
            draw_detections(frame, last_detections, color=(0, 255, 0), thickness=2, font_scale=0.65)
        frame_count += 1
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(1 / 15)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analysis", methods=["GET"])
def api_analysis():
    """GET ?enabled=1 to start analysis (detection + logging), ?enabled=0 to stop (plain livestream)."""
    global _analysis_enabled
    enabled = request.args.get("enabled")
    if enabled is not None:
        with _analysis_lock:
            _analysis_enabled = str(enabled).strip().lower() in ("1", "true", "yes")
    with _analysis_lock:
        return jsonify({"analysis": _analysis_enabled})


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    # 0.0.0.0 so you can open from another device on the network
    app.run(host="0.0.0.0", port=5000, threaded=True)
