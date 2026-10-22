"""
Minimal FastAPI inference backend for the brain-tumor MRI classifier demo.

Loads the retrained model (see ../models/brain_tumor_resnet50.keras) and the
Grad-CAM utilities, serves:
  - POST /predict  : upload an MRI image, get back predicted class + confidence
                      + a base64 PNG Grad-CAM overlay.
  - GET  /          : the static single-page upload-and-classify frontend.

This backend, the Grad-CAM module, and the retrained model are original work
added on top of the base pipeline from Masoud Nickparvar's
Brain-Tumor-MRI-Classification project (MIT License):
https://github.com/masoudnick/Brain-Tumor-MRI-Classification

Run with:
    uvicorn backend.app:app --reload --port 8000
(from the repo root, with the project's virtualenv active)
"""
from __future__ import annotations

import base64
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

from gradcam import gradcam_for_image  # noqa: E402
from model_utils import LABELS  # noqa: E402

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "brain_tumor_resnet50.keras"),
)

app = FastAPI(title="Brain Tumor MRI Classifier Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. Train it first with src/train.py."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


@app.get("/")
def index():
    static_index = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(static_index)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_path": MODEL_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    model = get_model()
    overlay, predicted_label, confidence, probs = gradcam_for_image(img, model)

    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode Grad-CAM overlay.")
    gradcam_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return JSONResponse({
        "predicted_class": predicted_label,
        "confidence": confidence,
        "probabilities": {label: float(p) for label, p in zip(LABELS, probs)},
        "gradcam_png_base64": gradcam_b64,
    })


# Serve the frontend's static assets (index.html references nothing external,
# but keep this mounted for favicon/etc. if added later).
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
