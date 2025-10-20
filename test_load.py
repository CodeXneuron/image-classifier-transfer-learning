# test_load.py
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = Path("binary_cnn_model.h5")

print("CWD:", Path.cwd())
print("Looking for model at:", MODEL_PATH.resolve())

try:
    m = load_model(MODEL_PATH)
    print("✅ Loaded via keras.models.load_model()")
except Exception as e:
    print("First load failed:", e)
    print("Trying fallback ...")
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Loaded via tf.keras.models.load_model(..., compile=False)")

print("Layers:", len(m.layers))
