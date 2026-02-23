"""
TFLite inference for YOLOv8-nano defect detection.
Loads best_float32.tflite (or best_float16.tflite) and labels.txt, runs on frames, returns boxes + classes.
"""

import os
from pathlib import Path

import cv2
import numpy as np

# Use TensorFlow's built-in TFLite interpreter (no tflite_runtime needed).
import tensorflow.lite as tflite

# Default paths (set MODEL_PATH and LABELS_PATH env or pass to DefectDetector)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "best_float32.tflite"
DEFAULT_LABELS_PATH = SCRIPT_DIR / "labels.txt"
MODEL_INPUT_SIZE = 640  # YOLOv8 export default
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45


class DefectDetector:
    """Run YOLOv8 TFLite on a frame; return list of (x1, y1, x2, y2, class_name, confidence)."""

    def __init__(self, model_path=None, labels_path=None):
        self.model_path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
        self.labels_path = Path(labels_path or os.environ.get("LABELS_PATH", DEFAULT_LABELS_PATH))
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Labels not found: {self.labels_path}")

        self.labels = self._load_labels()
        self.interpreter = tflite.Interpreter(model_path=str(self.model_path), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]
        self.input_h, self.input_w = self.input_shape[1], self.input_shape[2]
        self.input_dtype = self.input_details[0]["dtype"]

    def _load_labels(self):
        with open(self.labels_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def detect(self, frame, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
        """
        Run detection on a BGR frame (e.g. from OpenCV).
        Returns list of (x1, y1, x2, y2, class_name, confidence) in frame pixel coords.
        """
        h, w = frame.shape[:2]
        # Resize to model input, keep aspect ratio or letterbox; YOLOv8 often expects square
        inp = cv2.resize(frame, (self.input_w, self.input_h))
        if inp.ndim == 2:
            inp = cv2.cvtColor(inp, cv2.COLOR_GRAY2BGR)
        if self.input_dtype == np.float32:
            inp = inp.astype(np.float32) / 255.0
        else:
            inp = inp.astype(np.uint8)
        inp = np.expand_dims(inp, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()

        # Output shape is (1, 4+nc, 8400) or (1, 8400, 4+nc). We need (8400, 4+nc).
        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        if out.ndim == 3:
            out = out[0]
        # (10, 8400) -> (8400, 10); (8400, 10) already correct
        if out.shape[1] == 8400 and out.shape[0] != 8400:
            out = out.T
        elif out.shape[0] != 8400 and out.shape[-1] == 8400:
            out = np.transpose(out, (1, 0))
        nc = out.shape[1] - 4
        # Only use first 6 classes (our labels); model may export more
        nc = min(nc, len(self.labels))
        boxes_xywh = out[:, :4]
        scores = out[:, 4 : 4 + nc]

        # Class score: max over classes; class_id = argmax
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)
        if scores.dtype == np.float32 and np.any(confs > 2):
            # Logits -> sigmoid
            scores = 1.0 / (1.0 + np.exp(-scores))
            confs = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)

        mask = confs >= conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh (in 0–1 or input size) -> xyxy in input size then scale to frame
        xc, yc, bw, bh = boxes_xywh.T
        if np.max(xc) <= 1.0 and np.max(yc) <= 1.0:
            xc, yc, bw, bh = xc * self.input_w, yc * self.input_h, bw * self.input_w, bh * self.input_h
        x1 = (xc - bw / 2)
        y1 = (yc - bh / 2)
        x2 = (xc + bw / 2)
        y2 = (yc + bh / 2)
        # Scale to original frame size
        scale_x, scale_y = w / self.input_w, h / self.input_h
        x1 = (x1 * scale_x).astype(int)
        y1 = (y1 * scale_y).astype(int)
        x2 = (x2 * scale_x).astype(int)
        y2 = (y2 * scale_y).astype(int)
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confs.tolist(),
            conf_threshold,
            iou_threshold,
        )
        if len(indices) == 0:
            return []
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        out_list = []
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i]
            cid = int(class_ids[i])
            name = self.labels[cid] if cid < len(self.labels) else str(cid)
            out_list.append((int(x1), int(y1), int(x2), int(y2), name, float(confs[i])))
        return out_list


def draw_detections(frame, detections, color=(0, 255, 0), thickness=2, font_scale=0.6):
    """Draw bounding boxes and defect names + confidence on frame (BGR). Modifies frame in place."""
    for x1, y1, x2, y2, name, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA,
        )
    return frame
