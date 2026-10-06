"""On-demand, qualitative Grad-CAM explanations for the saved classifier."""
from functools import lru_cache
from io import BytesIO
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from backend.services.inference import _load, predict_image, predict_scores

logger = logging.getLogger(__name__)


class GradCAMConsistencyError(RuntimeError):
    """The explanation graph disagrees with the normal inference pipeline."""


@lru_cache(maxsize=1)
def _explanation_model():
    """Expose the saved model's backbone output and classifier in one graph."""
    import tensorflow as tf

    model, _ = _load()
    backbone = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                     and any(isinstance(child, tf.keras.layers.Conv2D) for child in layer.layers)), None)
    if backbone is None:
        raise RuntimeError("Saved classifier has no nested convolutional feature extractor")
    nodes = backbone._inbound_nodes
    if not nodes:
        raise RuntimeError("Saved feature extractor is not connected to the classifier graph")
    # Tap the nested model's output tensor from its call node in the original
    # Functional graph. Rebuilding the head layer-by-layer changed predictions
    # for this saved artifact, so preserve the original graph end to end.
    outer_backbone_output = nodes[-1].output_tensors
    if isinstance(outer_backbone_output, (list, tuple)):
        if len(outer_backbone_output) != 1:
            raise RuntimeError("Saved feature extractor has an unexpected output structure")
        outer_backbone_output = outer_backbone_output[0]
    grad_model = tf.keras.Model(model.inputs, [outer_backbone_output, model.outputs[0]], name="gradcam_model")
    return grad_model, backbone.name


def _render_overlay(original, heatmap):
    original = ImageOps.exif_transpose(original).convert("RGB")
    width, height = original.size
    resized = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize((width, height), Image.Resampling.BILINEAR)
    # Red/yellow heatmap blended with the original, with alpha rising only in
    # the most relevant regions so the crop remains visible underneath.
    h = np.asarray(resized, dtype=np.float32) / 255.0
    color = np.zeros((height, width, 3), dtype=np.float32)
    color[..., 0] = np.clip(2.0 * h, 0, 1)
    color[..., 1] = np.clip(2.0 - 2.0 * np.abs(h - 0.5) * 2.0, 0, 1)
    alpha = (0.15 + 0.45 * h)[..., None]
    base = np.asarray(original, dtype=np.float32)
    output = np.clip(base * (1.0 - alpha) + color * 255.0 * alpha, 0, 255).astype(np.uint8)
    if not np.isfinite(output).all():
        raise ValueError("Grad-CAM overlay contains invalid numeric values")
    buffer = BytesIO()
    Image.fromarray(output).save(buffer, format="PNG")
    return buffer.getvalue()


def generate_gradcam(image_path, expected_prediction, *, include_original=True, include_heatmap=True):
    """Return original and overlay PNG bytes for the authoritative prediction."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError("Analysis image is unavailable")
    model, config = _load()
    labels = config["labels"]
    normal_prediction = predict_image(image_path)
    if normal_prediction["prediction"] != expected_prediction:
        logger.error("Grad-CAM withheld: stored analysis prediction differs from current inference prediction")
        raise GradCAMConsistencyError("Analysis prediction does not match inference")
    expected_index = next((i for i, label in enumerate(labels)
                           if label.replace("___", " — ").replace("_", " ") == expected_prediction), None)
    if expected_index is None:
        raise ValueError("Saved prediction does not match the model label mapping")
    normal_scores, _ = predict_scores(image_path)
    inference_index = int(np.argmax(normal_scores))
    if inference_index != expected_index:
        raise GradCAMConsistencyError("Analysis prediction class does not match inference")
    height, width = config["image_size"]
    with Image.open(image_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
        model_image = original.resize((width, height))
        pixels = np.asarray(model_image, dtype=np.float32)[None, ...]
        original_buffer = BytesIO() if include_original else None
        if original_buffer is not None:
            original.save(original_buffer, format="PNG")
    try:
        import tensorflow as tf
        grad_model, _target_name = _explanation_model()
        input_tensor = tf.convert_to_tensor(pixels)
        with tf.GradientTape() as tape:
            feature_maps, scores = grad_model(input_tensor, training=False)
            scores = tf.convert_to_tensor(scores)
            if scores.shape.rank != 2 or int(tf.shape(scores)[0].numpy()) != 1:
                raise GradCAMConsistencyError("Grad-CAM returned invalid classifier scores")
            explanation_index = int(tf.argmax(scores[0], axis=-1).numpy())
            if explanation_index != expected_index:
                # The normal inference prediction remains authoritative. Never
                # produce a heatmap from a graph that predicts a different class.
                logger.error(
                    "Grad-CAM withheld: explanation class index %s differs from inference class index %s",
                    explanation_index, expected_index,
                )
                raise GradCAMConsistencyError("Grad-CAM prediction does not match inference")
            class_score = scores[:, expected_index]
        gradients = tape.gradient(class_score, feature_maps)
        if gradients is None:
            raise RuntimeError("Could not calculate gradients for the selected feature layer")
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        raw_map = tf.reduce_sum(feature_maps * weights, axis=-1)[0]
        raw_map = tf.nn.relu(raw_map)
        maximum = tf.reduce_max(raw_map)
        heatmap = tf.where(maximum > 0, raw_map / tf.maximum(maximum, tf.keras.backend.epsilon()),
                           tf.zeros_like(raw_map))
        heatmap = np.asarray(heatmap.numpy(), dtype=np.float32)
        if not np.isfinite(heatmap).all():
            raise ValueError("Grad-CAM heatmap contains invalid numeric values")
        heatmap = np.clip(heatmap, 0.0, 1.0)
        # A valid all-zero heatmap is preferable to fabricated activation.
        heatmap_buffer = BytesIO() if include_heatmap else None
        if heatmap_buffer is not None:
            Image.fromarray(np.uint8(heatmap * 255)).resize(original.size, Image.Resampling.BILINEAR).save(heatmap_buffer, format="PNG")
        overlay = _render_overlay(original, heatmap)
        return {"original_png": original_buffer.getvalue() if original_buffer else None, "overlay_png": overlay,
                "heatmap_png": heatmap_buffer.getvalue() if heatmap_buffer else None, "prediction": expected_prediction,
                "target_class_index": expected_index, "target_layer": _target_name}
    except Exception:
        logger.exception("Grad-CAM generation failed")
        raise
