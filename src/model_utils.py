"""
Shared preprocessing / model-building utilities for the brain tumor classifier.

The preprocessing pipeline (grayscale -> bilateral filter -> COLORMAP_BONE pseudo-color
-> resize) and the model architecture (frozen ResNet50 backbone + GlobalAveragePooling2D
+ Dropout + Dense softmax head) follow the approach demonstrated in the original
Brain-Tumor-MRI-Classification project by Masoud Nickparvar
(https://github.com/masoudnick/Brain-Tumor-MRI-Classification, MIT License).

This module is original code written for the retraining / Grad-CAM / demo work added
on top of that base project.
"""
from __future__ import annotations

import cv2
import numpy as np

LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 200


def preprocess_image(path_or_array, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Reproduce the notebook's preprocessing: grayscale read, bilateral filter,
    COLORMAP_BONE pseudo-coloring, resize to (image_size, image_size).

    Accepts either a file path (str) or an already-loaded BGR/gray numpy array.
    Returns a float32 array in [0, 1] of shape (image_size, image_size, 3).
    """
    if isinstance(path_or_array, str):
        image = cv2.imread(path_or_array, 0)
        if image is None:
            raise ValueError(f"Could not read image at {path_or_array}")
    else:
        arr = path_or_array
        if arr.ndim == 3:
            image = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            image = arr

    image = cv2.bilateralFilter(image, 2, 50, 50)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.resize(image, (image_size, image_size))
    return (image / 255.0).astype(np.float32)


def build_model(image_size: int = IMAGE_SIZE, num_classes: int = 4, freeze_base: bool = True):
    """Build the ResNet50 transfer-learning classifier.

    Architecture mirrors the original notebook's head (GlobalAveragePooling2D ->
    Dropout(0.4) -> Dense(num_classes, softmax)) on top of an ImageNet-pretrained
    ResNet50 backbone. The backbone is frozen by default to keep retraining fast
    and practical on commodity hardware (the original notebook left the backbone
    trainable end-to-end over 50 epochs; this is a deliberate, documented change --
    see README).
    """
    from tensorflow.keras.applications.resnet import ResNet50
    from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = not freeze_base

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=x)
    return model, base
