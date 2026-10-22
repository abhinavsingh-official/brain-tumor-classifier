"""Quick end-to-end smoke test: starts no server itself, expects the FastAPI
backend to already be running (e.g. `uvicorn backend.app:app --port 8000`),
and POSTs a real test image to /predict, printing the response.
"""
import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/predict"
image_path = sys.argv[2]

with open(image_path, "rb") as f:
    resp = requests.post(url, files={"file": (image_path, f, "image/jpeg")})

print("status:", resp.status_code)
data = resp.json()
print("predicted_class:", data.get("predicted_class"))
print("confidence:", data.get("confidence"))
print("probabilities:", data.get("probabilities"))
print("gradcam_png_base64 length:", len(data.get("gradcam_png_base64", "")))
