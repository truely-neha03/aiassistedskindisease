"""
app.py
-------
Gradio app: upload a skin image -> predicted disease NAME ONLY (no confidence
score shown), using the exact same preprocessing as training.

If the model isn't confident in any single class (top softmax probability
below CONFIDENCE_THRESHOLD), it returns "Unidentified" instead of forcing a
guess. Given this model's real validation accuracy (~65-70%, see
run_summary.json), forcing a class name on every image would be misleading
on cases it genuinely can't distinguish -- this makes that uncertainty
visible instead of hiding it.

Run:
    python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import json
import re

import gradio as gr
import numpy as np
import tensorflow as tf

MODEL_PATH = "skin_model_kaggle.h5"
CLASS_MAP_PATH = "class_names.json"
IMG_SIZE = (224, 224)

# Minimum top-class softmax probability required to return a named class.
# Below this, the app returns "Unidentified" rather than a low-confidence
# guess. Tune this against your own val set: a lower value shows more named
# guesses (some wrong), a higher value shows more "Unidentified" (fewer
# wrong guesses, more punts). 0.5 is a reasonable starting point for a
# 9-class softmax where random guessing would be ~0.11.
CONFIDENCE_THRESHOLD = 0.5
UNIDENTIFIED_LABEL = "Unidentified"

# IMPORTANT: this must be the SAME preprocess_input used in 03_train.py for
# whichever --model_name you actually trained with. If you trained with a
# different backbone, change this import to match (see the mapping below).
#
#   MobileNetV2     -> tf.keras.applications.mobilenet_v2.preprocess_input
#   EfficientNetB0   -> tf.keras.applications.efficientnet.preprocess_input
#   DenseNet121      -> tf.keras.applications.densenet.preprocess_input
#   ResNet50         -> tf.keras.applications.resnet50.preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)  # {"0": "class_a", "1": "class_b", ...}
# ensure ordering matches the model's output index order
class_names = [class_map[str(i)] for i in range(len(class_map))]


def clean(name):
    """'1. Eczema 1677' -> 'Eczema' (display only; index order is unchanged)."""
    name = re.sub(r"^\d+\.\s*", "", name)
    name = re.sub(r"\s*-?\s*\d+(\.\d+)?k?$", "", name)
    return name.strip()


display_names = [clean(n) for n in class_names]


def predict(image):
    if image is None:
        return "No image provided."

    img = tf.image.resize(image, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    arr = np.expand_dims(img.numpy(), axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_prob = float(preds[top_idx])

    if top_prob < CONFIDENCE_THRESHOLD:
        return UNIDENTIFIED_LABEL
    return display_names[top_idx]


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a skin image"),
    outputs=gr.Textbox(label="Predicted condition"),
    title="Skin Disease Detection",
    description=(
        "Upload a clear, well-lit image of the affected skin area. "
        "This is a demo model trained on a Kaggle dataset for a learning "
        "project — it is NOT a medical device and must not be used for "
        "real diagnosis. Consult a dermatologist for any actual concern."
    ),
)

if __name__ == "__main__":
    demo.launch()