"""
Grad-CAM explainability for the retrained brain-tumor classifier.

Standard Keras/TF Grad-CAM recipe (Selvaraju et al., 2017) applied against the
frozen ResNet50 backbone's last convolutional feature map. This module is
original code added on top of the base project by Masoud Nickparvar
(https://github.com/masoudnick/Brain-Tumor-MRI-Classification, MIT License).
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from model_utils import LABELS, preprocess_image

LAST_CONV_LAYER_NAME = "conv5_block3_out"  # last conv block of ResNet50


def make_gradcam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: str = LAST_CONV_LAYER_NAME,
                          pred_index: int | None = None):
    """img_array: shape (1, H, W, 3), already preprocessed/normalized."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def overlay_heatmap(original_bgr_float: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """original_bgr_float: (H, W, 3) float in [0,1] (the already-preprocessed pseudo-colored image).
    Returns an overlay image as uint8 BGR, suitable for cv2.imwrite / PNG encoding.
    """
    h, w = original_bgr_float.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    base_uint8 = np.uint8(255 * original_bgr_float)
    overlay = cv2.addWeighted(heatmap_color, alpha, base_uint8, 1 - alpha, 0)
    return overlay


def gradcam_for_image(path_or_array, model, image_size: int = 200, pred_index: int | None = None):
    """End-to-end: preprocess -> predict -> Grad-CAM -> overlay.
    Returns (overlay_bgr_uint8, predicted_label, confidence, probs).
    """
    processed = preprocess_image(path_or_array, image_size)
    img_array = np.expand_dims(processed, axis=0)

    heatmap, pred_idx, probs = make_gradcam_heatmap(img_array, model, pred_index=pred_index)
    overlay = overlay_heatmap(processed, heatmap)

    predicted_label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])
    return overlay, predicted_label, confidence, probs
