"""
Retrain the brain-tumor MRI classifier on the real Kaggle dataset
(masoudnickparvar/brain-tumor-mri-dataset) using the preprocessing + architecture
approach from the original Brain-Tumor-MRI-Classification project by Masoud
Nickparvar (https://github.com/masoudnick/Brain-Tumor-MRI-Classification, MIT
License).

This script is original code: it adapts the notebook's pipeline into a
reusable, reduced-epoch training run that actually produces and saves a model
file, plus a JSON file of real training/evaluation metrics.

Usage:
    python src/train.py --data-dir /path/to/dataset --epochs 15
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from model_utils import LABELS, IMAGE_SIZE, build_model, preprocess_image


def load_split(data_dir: str, split: str):
    xs, ys = [], []
    for label in LABELS:
        split_path = os.path.join(data_dir, split, label)
        files = sorted(os.listdir(split_path))
        for fname in tqdm(files, desc=f"{split}/{label}"):
            fpath = os.path.join(split_path, fname)
            xs.append(preprocess_image(fpath, IMAGE_SIZE))
            ys.append(LABELS.index(label))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing Training/ and Testing/ subfolders "
                             "(each with glioma/meningioma/notumor/pituitary subfolders)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    parser.add_argument("--metrics-out", default=os.path.join(os.path.dirname(__file__), "..", "models", "training_metrics.json"))
    parser.add_argument("--freeze-base", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import tensorflow as tf
    if os.environ.get("FORCE_CPU") == "1":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception as e:
            print("Could not disable GPU:", e)
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical

    print("Loading & preprocessing training images ...")
    t0 = time.time()
    x_train_full, y_train_full = load_split(args.data_dir, "Training")
    x_test, y_test = load_split(args.data_dir, "Testing")
    print(f"Loaded {len(x_train_full)} training + {len(x_test)} testing images in {time.time()-t0:.1f}s")
    print("x_train_full shape:", x_train_full.shape, "x_test shape:", x_test.shape)

    x_train_full, y_train_full = shuffle(x_train_full, y_train_full, random_state=42)
    y_train_full_oh = to_categorical(y_train_full, num_classes=len(LABELS))
    y_test_oh = to_categorical(y_test, num_classes=len(LABELS))

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full_oh, test_size=0.2, random_state=42
    )
    print("x_train:", x_train.shape, "x_val:", x_val.shape)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    model, base = build_model(IMAGE_SIZE, len(LABELS), freeze_base=args.freeze_base)
    adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    ckpt_path = os.path.join(args.out_dir, "brain_tumor_resnet50_best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5,
                      mode="min", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1, mode="min"),
        ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", verbose=1,
                        save_best_only=True, mode="min"),
    ]

    print(f"Training for up to {args.epochs} epochs (frozen_base={args.freeze_base}) ...")
    t_train0 = time.time()
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
    )
    train_time = time.time() - t_train0
    print(f"Training finished in {train_time:.1f}s")

    final_model_path = os.path.join(args.out_dir, "brain_tumor_resnet50.keras")
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")

    test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=1)
    predicted_classes = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, predicted_classes)
    report = classification_report(y_test, predicted_classes, target_names=LABELS, output_dict=True)
    report_txt = classification_report(y_test, predicted_classes, target_names=LABELS)
    print(report_txt)
    print("Test accuracy:", test_acc, "Test loss:", test_loss)

    epochs_ran = len(history.history["loss"])
    best_val_acc = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])

    metrics = {
        "epochs_requested": args.epochs,
        "epochs_ran": epochs_ran,
        "batch_size": args.batch_size,
        "image_size": IMAGE_SIZE,
        "frozen_base": args.freeze_base,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "train_time_seconds": train_time,
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classification_report_text": report_txt,
        "labels": LABELS,
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
