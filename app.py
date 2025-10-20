import os
from io import BytesIO
from base64 import b64encode
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from flask import (
    Flask, request, redirect, url_for,
    jsonify, render_template, render_template_string
)

# ---------------- Config ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "binary_cnn_model.h5")
TARGET_SIZE = (128, 128)             # trained on 128x128
CLASS0 = os.getenv("CLASS0", "Cat")
CLASS1 = os.getenv("CLASS1", "Dog")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# Limit uploaded file size (10 MB)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ---------------- Model (lazy load) ----------------
_model = None
def get_model():
    """Load the model on first use to avoid slow dyno boot."""
    global _model
    if _model is None:
        # Import TensorFlow only when needed (saves memory/boot time)
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        try:
            _model = load_model(MODEL_PATH)
        except Exception:
            # compile=False for old Keras formats
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model

# ---------------- Utils ----------------
def preprocess_image(pil_img, target_size=TARGET_SIZE):
    img = pil_img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

def run_predict(arr):
    model = get_model()
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS1 if prob >= THRESHOLD else CLASS0
    return prob, label

def pil_to_dataurl(pil_img):
    preview = pil_img.copy()
    preview.thumbnail((720, 720))
    buf = BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    b64 = b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def template_exists(name: str) -> bool:
    return Path(app.root_path, "templates", name).exists()

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return "ok", 200

@app.get("/")
def index():
    ctx = dict(
        title="Binary Image Classifier",
        class0=CLASS0, class1=CLASS1,
        threshold=int(THRESHOLD * 100),
        target_size=TARGET_SIZE,
    )
    if template_exists("index.html"):
        return render_template("index.html", **ctx)
    # Minimal inline page if you don't have templates yet
    return render_template_string(
        """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
          <title>{{ title }}</title>
        </head>
        <body class="bg-light">
          <div class="container py-5">
            <h1 class="mb-3">{{ title }}</h1>
            <p class="text-muted">Classes: <b>{{ class0 }}</b> vs <b>{{ class1 }}</b> • Threshold: {{ threshold }}%</p>
            <form action="{{ url_for('predict_route') }}" method="post" enctype="multipart/form-data" class="mt-4">
              <input class="form-control" type="file" name="image" accept="image/*" required>
              <button class="btn btn-primary mt-3" type="submit">Predict</button>
            </form>
          </div>
        </body>
        </html>
        """,
        **ctx,
    )

@app.post("/predict")
def predict_route():
    if "image" not in request.files or request.files["image"].filename == "":
        return redirect(url_for("index"))

    file = request.files["image"]
    pil_img = Image.open(BytesIO(file.read()))
    arr = preprocess_image(pil_img)
    prob, label = run_predict(arr)
    prob_pct = round(prob * 100, 2)
    is_positive = (label == CLASS1)
    badge_class = "text-bg-success" if is_positive else "text-bg-secondary"
    bar_class = "bg-success" if is_positive else "bg-secondary"

    ctx = dict(
        title="Prediction Result",
        filename=file.filename,
        label=label,
        prob_pct=prob_pct,
        threshold=int(THRESHOLD * 100),
        badge_class=badge_class,
        bar_class=bar_class,
        preview_dataurl=pil_to_dataurl(pil_img),
    )
    if template_exists("result.html"):
        return render_template("result.html", **ctx)

    # Minimal inline result page
    return render_template_string(
        """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
          <title>{{ title }}</title>
        </head>
        <body class="bg-light">
          <div class="container py-5">
            <a href="{{ url_for('index') }}" class="btn btn-link">&larr; Back</a>
            <h2 class="mt-3">Result for <code>{{ filename }}</code></h2>
            <span class="badge {{ badge_class }} fs-6">{{ label }}</span>
            <div class="progress my-3" role="progressbar" aria-valuenow="{{ prob_pct }}" aria-valuemin="0" aria-valuemax="100" style="height: 22px;">
              <div class="progress-bar {{ bar_class }}" style="width: {{ prob_pct }}%">{{ prob_pct }}%</div>
            </div>
            <img class="img-fluid rounded border" src="{{ preview_dataurl }}" alt="preview">
          </div>
        </body>
        </html>
        """,
        **ctx,
    )

@app.post("/api/predict")
def api_predict():
    # raw bytes (image/*) in body
    pil_img = Image.open(BytesIO(request.data))
    arr = preprocess_image(pil_img)
    prob, label = run_predict(arr)
    return jsonify({"prob": prob, "label": label, "threshold": THRESHOLD})
