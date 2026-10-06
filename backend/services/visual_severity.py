"""Conservative color-based visual damage indicator; not calibrated severity."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SEVERITY_METHOD = "image_based_visual_estimation"
SEVERITY_EXPLANATION = (
    "Severity is an AI-assisted visual estimate based on visible affected regions "
    "in the uploaded image. It is not an official PMFBY loss percentage."
)


def _rgb_to_hsv(rgb):
    pixels = rgb.astype(np.float32) / 255.0
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    maximum = pixels.max(axis=2)
    minimum = pixels.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red_max = (maximum == red) & nonzero
    green_max = (maximum == green) & nonzero
    blue_max = (maximum == blue) & nonzero
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4
    hue *= 60
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-6)
    return hue, saturation, maximum


def _foreground_mask(rgb, saturation, value):
    height, width = rgb.shape[:2]
    border_width = max(2, min(height, width) // 16)
    border = np.concatenate((
        rgb[:border_width].reshape(-1, 3), rgb[-border_width:].reshape(-1, 3),
        rgb[:, :border_width].reshape(-1, 3), rgb[:, -border_width:].reshape(-1, 3),
    ), axis=0)
    background = np.median(border, axis=0)
    distance = np.sqrt(np.sum((rgb.astype(np.float32) - background) ** 2, axis=2))
    mask = (distance >= 35) & (saturation >= 0.10) & (value >= 0.12)
    coverage = float(mask.mean())
    reliable = 0.05 <= coverage <= 0.85
    if not reliable:
        # Background contrast can fail on busy field photos; use chromatic pixels
        # as a fallback and mark the result as less reliable in its explanation.
        mask = (saturation >= 0.18) & (value >= 0.12)
    return mask, reliable


def _category(percent):
    if percent <= 5.0:
        return "Low"
    if percent <= 20.0:
        return "Moderate"
    return "Severe"


def estimate_visual_damage(image_path: str | Path, healthy: bool):
    """Estimate chromatic affected area; output is not calibrated or validated."""
    if healthy:
        return {
            "severity": "Healthy",
            "estimated_visual_damage_indicator": 0.0,
            "severity_method": SEVERITY_METHOD,
            "severity_explanation": SEVERITY_EXPLANATION,
        }

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((512, 512))
        rgb = np.asarray(image, dtype=np.uint8)
    hue, saturation, value = _rgb_to_hsv(rgb)
    leaf_mask, segmentation_reliable = _foreground_mask(rgb, saturation, value)
    leaf_pixels = int(leaf_mask.sum())
    if leaf_pixels:
        # Neutral background, low-saturation highlights, and green-to-green
        # variation are excluded; only saturated yellow/orange/brown colors count.
        affected = ((hue <= 42) | (hue >= 350)) & (saturation >= 0.22) & (value >= 0.12)
        percent = round(100.0 * int(np.count_nonzero(affected & leaf_mask)) / leaf_pixels, 1)
    else:
        percent = 0.0
    explanation = SEVERITY_EXPLANATION
    if not segmentation_reliable:
        explanation += " Foreground segmentation was uncertain, so treat the estimate with extra caution."
    return {
        "severity": _category(percent),
        "estimated_visual_damage_indicator": percent,
        "severity_method": SEVERITY_METHOD,
        "severity_explanation": explanation,
    }
