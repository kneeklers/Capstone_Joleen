"""
Train YOLOv8-nano on NEU-DET (6 defect classes) and export to TFLite for Pi5.
Use when you need to detect multiple defect types and multiple defects per image.

Prerequisite: Convert NEU-DET to YOLO format first:
  python convert_neu_det_to_yolo.py --input-dir /content/NEU-DET --output-dir /content/yolo_dataset

Then:
  python train_yolo.py --data-dir /content/yolo_dataset --output-dir /content/output
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-nano for defect detection, export TFLite")
    parser.add_argument("--data-dir", type=str, required=True, help="YOLO dataset root (contains data.yaml)")
    parser.add_argument("--output-dir", type=str, default="/content/output", help="Save runs and exported model here")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size (640 or 320 for lighter Pi)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    data_yaml = os.path.join(args.data_dir, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}. Run convert_neu_det_to_yolo.py first.")

    os.makedirs(args.output_dir, exist_ok=True)

    # YOLOv8-nano: best speed/size for Pi5 while keeping good accuracy
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.output_dir,
        name="yolov8n_defect",
        exist_ok=True,
        pretrained=True,
    )

    # Export to TFLite for Raspberry Pi
    best_pt = Path(args.output_dir) / "yolov8n_defect" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Best weights not found at {best_pt}")

    export_model = YOLO(str(best_pt))
    export_model.export(
        format="tflite",
        imgsz=args.imgsz,
        half=False,  # FP32 for Pi if no FP16 support; set True to try smaller model
        int8=False,
        optimize=True,
    )
    # Ultralytics writes .tflite next to best.pt (e.g. best_float32.tflite)
    export_dir = best_pt.parent
    tflite_files = list(Path(export_dir).glob("*.tflite"))
    if tflite_files:
        import shutil
        for src in tflite_files:
            dst = Path(args.output_dir) / src.name
            shutil.copy2(src, dst)
            print("TFLite copied to", dst)
    else:
        print("TFLite export path:", export_dir)

    # Labels for inference on Pi (same order as data.yaml)
    labels_path = Path(args.output_dir) / "labels.txt"
    with open(labels_path, "w") as f:
        f.write("\n".join(CLASS_NAMES))
    print("Labels saved to", labels_path)

    return results


if __name__ == "__main__":
    main()
