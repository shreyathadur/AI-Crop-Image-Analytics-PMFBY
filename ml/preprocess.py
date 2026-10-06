"""Shared Keras image preprocessing helpers."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ml.config import IMAGE_SIZE


def load_rgb_image(path: str | Path, size=IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB").resize((size[1], size[0]))
        return np.asarray(image, dtype=np.float32)


def write_preprocessing_config(path: Path, labels: list[str], size=IMAGE_SIZE):
    path.write_text(json.dumps({"image_size": list(size), "labels": labels,
                                "preprocessing": "MobileNetV2 preprocess_input; RGB uint8 resized to configured size"}, indent=2), encoding="utf-8")
