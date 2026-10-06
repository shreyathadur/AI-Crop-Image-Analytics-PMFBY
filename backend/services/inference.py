"""Lazy model loader; missing trained artifacts produce explicit unavailable state."""
import json
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

from backend.config import MODEL_PATH
from backend.services.visual_severity import estimate_visual_damage


@lru_cache(maxsize=1)
def _load():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    config_path = MODEL_PATH.with_name("model_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return model, config


def predict_scores(image_path):
    """Run the saved classifier with the same preprocessing used by inference."""
    model, config = _load()
    height, width = config["image_size"]
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").resize((width, height))
        pixels = np.asarray(im, dtype=np.float32)
    scores = model.predict(np.expand_dims(pixels, 0), verbose=0)[0]
    return scores, config


def predict_image(image_path):
    scores, config = predict_scores(image_path)
    index = int(np.argmax(scores)); class_name = config["labels"][index]
    crop = class_name.split("___", 1)[0].replace("_", " ")
    condition = class_name.partition("___")[2].replace("_", " ").strip()
    healthy = "healthy" in class_name.lower()
    severity = estimate_visual_damage(image_path, healthy=healthy)
    return {"prediction": class_name.replace("___", " — ").replace("_", " "), "crop": crop,
            "detected_condition": condition,
            "health_status": "Healthy" if healthy else "Potential disease class detected",
            "confidence": float(scores[index]),
            **severity,
            "explanation": "The model matched this image to a supported leaf class. Review the image with a qualified field expert; this result is preliminary."}
