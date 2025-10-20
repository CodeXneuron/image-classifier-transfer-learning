import os
from io import BytesIO
from base64 import b64encode
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf

# -------- Config --------
MODEL_PATH = os.getenv("MODEL_PATH", "binary_cnn_model.h5")
TARGET_SIZE = (128, 128)   # trained on 128x128
CLASS0 = os.getenv("CLASS0", "Cat")
CLASS1 = os.getenv("CLASS1", "Dog")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# Flask app + limits
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# -------- Model load --------
def load_cnn_model(path: str):
    try:
        return load_model(path)
    except Exception:
        return tf.keras.models.load_model(path, compile=False)

model = load_cnn_model(MODEL_PATH)

# -------- Preprocess --------
def preprocess_image(pil_img, target_size=TARGET_SIZE):
    img = pil_img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(arr):
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS1 if prob >= THRESHOLD else CLASS0
    return prob, label

def pil_to_dataurl(pil_img):
    """Return a small inline preview for the result page."""
    preview = pil_img.copy()
    preview.thumbnail((720, 720))
    buf = BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    b64 = b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# -------- Routes --------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        title="Binary Image Classifier",
        class0=CLASS0,
        class1=CLASS1,
        threshold=int(THRESHOLD*100),
        target_size=TARGET_SIZE,
    )

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files or request.files["image"].filename == "":
        return redirect(url_for("index"))

    file = request.files["image"]
    pil_img = Image.open(BytesIO(file.read()))
    arr = preprocess_image(pil_img)
    prob, label = predict(arr)
    prob_pct = round(prob * 100, 2)

    # choose colors
    is_positive = (label == CLASS1)
    badge_class = "text-bg-success" if is_positive else "text-bg-secondary"
    bar_class = "bg-success" if is_positive else "bg-secondary"

    return render_template(
        "result.html",
        title="Prediction Result",
        filename=file.filename,
        label=label,
        prob_pct=prob_pct,
        threshold=int(THRESHOLD*100),
        badge_class=badge_class,
        bar_class=bar_class,
        preview_dataurl=pil_to_dataurl(pil_img)
    )

# Optional JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    pil_img = Image.open(BytesIO(request.data))
    arr = preprocess_image(pil_img)
    prob, label = predict(arr)
    return jsonify({"prob": prob, "label": label, "threshold": THRESHOLD})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
