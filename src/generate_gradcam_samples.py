"""
Generate a handful of real Grad-CAM heatmap overlays (one or two per class) from
the Testing split, using the retrained model. Saves PNGs to gradcam_samples/.
"""
from __future__ import annotations

import argparse
import os

import cv2

from gradcam import gradcam_for_image
from model_utils import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing the Testing/ subfolder")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "gradcam_samples"))
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model_path)

    for label in LABELS:
        class_dir = os.path.join(args.data_dir, "Testing", label)
        files = sorted(os.listdir(class_dir))[: args.per_class]
        for i, fname in enumerate(files):
            fpath = os.path.join(class_dir, fname)
            overlay, pred_label, confidence, probs = gradcam_for_image(fpath, model)
            correct = "correct" if pred_label == label else "WRONG"
            out_name = f"{label}_{i}_true-{label}_pred-{pred_label}_{confidence:.2f}_{correct}.png"
            out_path = os.path.join(args.out_dir, out_name)
            cv2.imwrite(out_path, overlay)
            print(f"{fpath} -> true={label} pred={pred_label} conf={confidence:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
