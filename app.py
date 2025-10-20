import os
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf

# ------------ Config ------------
MODEL_PATH = os.getenv("MODEL_PATH", "binary_cnn_model.h5")
TARGET_SIZE = (128, 128)   # trained on 128x128
CLASS0 = os.getenv("CLASS0", "Cat")
CLASS1 = os.getenv("CLASS1", "Dog")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# ------------ Load model ------------
def load_cnn_model(path: str):
    try:
        return load_model(path)
    except Exception:
        return tf.keras.models.load_model(path, compile=False)

model = load_cnn_model(MODEL_PATH)

# ------------ Preprocess ------------
def preprocess_image(pil_img, target_size=TARGET_SIZE):
    img = pil_img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(arr):
    prob = float(model.predict(arr, verbose=0)[0][0])   # class_1 prob
    pred_is_one = prob >= THRESHOLD
    label = CLASS1 if pred_is_one else CLASS0
    return {"prob_class_1": prob, "label": label, "threshold": THRESHOLD}

# ------------ Flask app ------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", class0=CLASS0, class1=CLASS1, threshold=int(THRESHOLD*100))

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    pil_img = Image.open(BytesIO(file.read()))
    arr = preprocess_image(pil_img)
    out = predict(arr)

    # Render result page
    return render_template(
        "result.html",
        filename=file.filename,
        label=out["label"],
        prob_pct=round(out["prob_class_1"]*100, 2),
        class0=CLASS0,
        class1=CLASS1,
        threshold=int(THRESHOLD*100),
    )

# Optional JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    # expects raw image bytes
    pil_img = Image.open(BytesIO(request.data))
    arr = preprocess_image(pil_img)
    out = predict(arr)
    return jsonify(out)

if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=8000, debug=True)
