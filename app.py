"""
Live defect detection web app for Raspberry Pi 5.
Serves a simple page with MJPEG camera stream. Model inference can be added later.

Run: python app.py
Then open http://localhost:5000 (or http://<pi-ip>:5000 from another device)
"""

import io
import time
from threading import Lock

import cv2
from flask import Flask, Response, render_template

app = Flask(__name__)

# Camera: use 0 for default (Pi camera module or USB). Change if needed.
CAMERA_INDEX = 0
# Optional: set to (width, height) to fix resolution (e.g. (640, 480) for lighter load)
FRAME_SIZE = (640, 480)

_camera_lock = Lock()
_camera = None


def get_camera():
    """Lazy-init camera so we don't block at import."""
    global _camera
    with _camera_lock:
        if _camera is None:
            _camera = cv2.VideoCapture(CAMERA_INDEX)
            if FRAME_SIZE:
                _camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
                _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        return _camera


def generate_frames():
    """Yield JPEG frames for MJPEG stream. No model inference yet."""
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break
        # Optional: flip for mirror view
        # frame = cv2.flip(frame, 1)
        # TODO: run model here and draw boxes on frame
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(1 / 15)  # ~15 FPS to reduce CPU on Pi


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
