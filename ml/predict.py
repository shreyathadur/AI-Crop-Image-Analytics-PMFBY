"""Command-line inference for a saved crop classifier."""
import argparse
import json

import numpy as np
import tensorflow as tf

from ml.config import IMAGE_SIZE, MODEL_DIR
from ml.preprocess import load_rgb_image


def predict(image_path):
    config = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(MODEL_DIR / "crop_disease.keras")
    image = load_rgb_image(image_path, tuple(config["image_size"]))
    scores = model.predict(np.expand_dims(image, 0), verbose=0)[0]
    index = int(np.argmax(scores)); name = config["labels"][index]
    crop, _, status = name.partition("___")
    healthy = "healthy" in name.lower()
    return {"class_name": name, "crop": crop.replace("_", " "), "health_status": "Healthy" if healthy else "Affected; disease class predicted", "confidence": float(scores[index])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("image"); args = parser.parse_args()
    print(json.dumps(predict(args.image), indent=2))
