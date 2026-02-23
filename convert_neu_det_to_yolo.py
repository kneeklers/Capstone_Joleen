"""
Convert NEU-DET (Pascal VOC XML) to YOLO format for YOLOv8 training.
Run once after unzipping archive-2.zip (e.g. to /content/NEU-DET).

Usage:
  python convert_neu_det_to_yolo.py --input-dir /content/NEU-DET --output-dir /content/yolo_dataset
  # Then train: python train_yolo.py --data-dir /content/yolo_dataset
"""

import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# NEU-DET defect classes (order defines class_id 0, 1, ...)
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def parse_voc_xml(xml_path: str):
    """Parse Pascal VOC XML; return list of (class_name, xmin, ymin, xmax, ymax)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        objects.append((name, xmin, ymin, xmax, ymax, w, h))
    return objects


def voc_to_yolo_line(class_name: str, xmin: float, ymin: float, xmax: float, ymax: float, img_w: float, img_h: float) -> str:
    """Convert one bbox to YOLO line: class_id x_center y_center width height (normalized 0-1)."""
    cid = CLASS_TO_ID.get(class_name)
    if cid is None:
        raise ValueError(f"Unknown class: {class_name}. Known: {list(CLASS_TO_ID)}")
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main():
    parser = argparse.ArgumentParser(description="NEU-DET (VOC XML) to YOLO format")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to NEU-DET folder (contains IMAGES + ANNOTATIONS)")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write YOLO dataset (train/val + data.yaml)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of images for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = os.path.join(args.input_dir, "IMAGES")
    ann_dir = os.path.join(args.input_dir, "ANNOTATIONS")
    if not os.path.isdir(images_dir) or not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Expected IMAGES/ and ANNOTATIONS/ under {args.input_dir}")

    # Collect all image base names (no extension) that have both image and annotation
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    candidates = set()
    for f in os.listdir(images_dir):
        base, ext = os.path.splitext(f)
        if ext.lower() in image_exts:
            candidates.add((base, os.path.join(images_dir, f)))
    ann_by_base = {}
    for f in os.listdir(ann_dir):
        if not f.endswith(".xml"):
            continue
        base = os.path.splitext(f)[0]
        ann_by_base[base] = os.path.join(ann_dir, f)

    pairs = []
    for base, img_path in candidates:
        if base not in ann_by_base:
            continue
        pairs.append((base, img_path, ann_by_base[base]))

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_set = set(p[0] for p in pairs[:n_val])
    train_set = set(p[0] for p in pairs[n_val:])

    out = Path(args.output_dir)
    train_images = out / "train" / "images"
    train_labels = out / "train" / "labels"
    val_images = out / "val" / "images"
    val_labels = out / "val" / "labels"
    for d in (train_images, train_labels, val_images, val_labels):
        d.mkdir(parents=True, exist_ok=True)

    for base, img_path, xml_path in pairs:
        is_val = base in val_set
        if is_val:
            img_dst_dir = val_images
            lbl_dst_dir = val_labels
        else:
            img_dst_dir = train_images
            lbl_dst_dir = train_labels

        ext = os.path.splitext(img_path)[1]
        dst_img = img_dst_dir / f"{base}{ext}"
        dst_lbl = lbl_dst_dir / f"{base}.txt"

        # Copy image (or symlink to avoid duplicating; copy is safer on Colab)
        if not dst_img.exists() or os.path.getmtime(img_path) > os.path.getmtime(dst_img):
            import shutil
            shutil.copy2(img_path, dst_img)

        objs = parse_voc_xml(xml_path)
        lines = []
        for t in objs:
            name, xmin, ymin, xmax, ymax, w, h = t
            lines.append(voc_to_yolo_line(name, xmin, ymin, xmax, ymax, w, h))
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))

    # dataset.yaml for YOLOv8
    data_yaml = out / "data.yaml"
    abs_out = str(out.resolve())
    yaml_content = f"""# NEU-DET surface defect dataset (YOLO format)
path: {abs_out}
train: train/images
val: val/images

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches

nc: 6
"""
    with open(data_yaml, "w") as f:
        f.write(yaml_content)

    print(f"Converted {len(train_set)} train, {len(val_set)} val. Dataset YAML: {data_yaml}")
    print("Class names:", CLASS_NAMES)


if __name__ == "__main__":
    main()
