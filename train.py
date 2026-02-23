"""
Defect detection model training for Pi5 deployment.
Train on Colab/Kaggle (or any GPU runtime), export to TFLite for Raspberry Pi 5.

Dataset: folder-based. After downloading from Kaggle, organize as:
  <DATA_DIR>/
    train/
      defect/   (or your class names, e.g. defective, bad)
      good/     (e.g. ok, normal, non_defective)
    val/
      defect/
      good/

If your Kaggle dataset has different structure, either:
- Create train/val subdirs and symlink/copy, or
- Set DATA_DIR to the folder that already contains train/ and val/ with class subdirs.
"""

import os
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------------
# Config (override via args or env for Colab)
# -----------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "/content/data")  # Colab: /content/your_dataset
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/content/output")
MODEL_NAME = "defect_mobilenetv2"


def build_model(num_classes: int, img_size: int = IMG_SIZE):
    """
    Transfer learning with MobileNetV2. Lightweight and TFLite-friendly for Pi5.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze first; optional: unfreeze last layers later

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Lambda(keras.applications.mobilenet_v2.preprocess_input)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_datasets(data_dir: str, img_size: int, batch_size: int):
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    if not os.path.isdir(train_path):
        raise FileNotFoundError(
            f"Train folder not found: {train_path}. "
            "Create data/train/<class_name>/ and data/val/<class_name>/ with images."
        )

    train_ds = keras.utils.image_dataset_from_directory(
        train_path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        label_mode="int",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        label_mode="int",
    )
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train defect classifier and export TFLite")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Root dir containing train/ and val/")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Where to save SavedModel and .tflite")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Input size (square)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_ds, val_ds = get_datasets(data_dir, args.img_size, args.batch_size)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model(num_classes, args.img_size)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_weights.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Save full model (for TFLite conversion)
    saved_model_path = os.path.join(output_dir, MODEL_NAME)
    model.save(saved_model_path, save_format="tf")
    print("Saved model to", saved_model_path)

    # Export class names for inference on Pi
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(class_names))
    print("Labels saved to", labels_path)

    # Convert to TFLite (lightweight for Pi5)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(output_dir, f"{MODEL_NAME}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("TFLite model saved to", tflite_path)

    return saved_model_path, tflite_path, labels_path


if __name__ == "__main__":
    main()
