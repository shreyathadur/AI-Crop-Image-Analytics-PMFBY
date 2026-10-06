# ML model plan

## Choice

Use Keras MobileNetV2 with ImageNet initialization and a new 38-class softmax head. The model is compact and widely supported, making it a practical CPU inference choice; input size and batch size will be configurable. This implementation freezes the pretrained backbone and trains the new classification head.

## Training and artifacts

Use a fixed random seed, stratified held-out splits, training-only augmentation, early stopping, and best-validation checkpointing. Save the trained model, class-label mapping, image size and preprocessing configuration, and training history in `ml/models/` and `ml/results/`. Do not substitute a placeholder model when weights are absent: inference must state that a trained artifact is unavailable.

## Evaluation

The actual local run used 100 images per class, a 3,040/380/380 train/validation/test split, seed 42, and 3 epochs. On the 380-image held-out test split it measured 85.53% accuracy, 0.8643 macro precision, 0.8553 macro recall, and 0.8509 macro F1. Each class has only 10 test images, and this small, controlled-dataset run is not evidence of field accuracy. Per-class scores, confusion matrix, metrics, training curves, and report are in `ml/results/`.

## Visual severity indicator and interpretation

PlantVillage provides disease/healthy class labels, not severity or crop-loss labels. The disease classifier is unchanged. Its training graph applies `tf.keras.applications.mobilenet_v2.preprocess_input` after augmentation, so inference supplies EXIF-corrected RGB pixels resized to 224x224 in the original 0-255 range; preprocessing is already inside the saved model and must not be applied a second time.

For a class predicted as healthy, the API reports category `Healthy` and an indicator of 0.0. For affected classes, a separate deterministic image heuristic resizes a copy to at most 512x512, estimates foreground from color distance to the image border plus HSV saturation/value, and counts saturated yellow/orange/brown pixels within that foreground. Neutral background, low-saturation highlights, and green-to-green variation are excluded. Foreground segmentation falls back to a saturation mask when the contrast estimate is outside the expected 5-85% image coverage. A fallback result is flagged as especially uncertain in its explanation.

The resulting pixel ratio is named **Estimated Visual Damage Indicator**. The prototype maps 0-5% to Low, above 5-20% to Moderate, and above 20% to Severe. A disease-class prediction with no matching color pixels remains Low rather than being called healthy. These are transparent engineering bins, not learned or calibrated thresholds. The heuristic can miss pale mildew, dark lesions, chlorosis, shadows, and damage outside the leaf, and can count natural yellow/brown color. It is not agronomically validated. No severity accuracy is reported because there are no ground-truth severity labels. The indicator is never an official PMFBY damage percentage, crop-loss certification, or government decision.

## Grad-CAM explanation

The saved `ml/models/crop_disease.keras` artifact was inspected without changing or retraining it. It is a Keras Functional model with a nested `mobilenetv2_1.00_224` feature extractor, `(None, 224, 224, 3)` input, and 38-class output. Grad-CAM taps the backbone's 7x7x1280 spatial output directly from the saved model's original Functional graph, then differentiates the original classifier score with respect to that activation. This keeps the explanation output and feature map connected through the same forward graph.

Explanations are generated on demand with the same EXIF correction, RGB conversion, 224x224 resize, and in-model MobileNetV2 preprocessing as normal inference. Grad-CAM first reruns the existing inference pipeline and requires its predicted class to match the saved analysis prediction; it then checks that the explanation graph's argmax is the same class before calculating gradients. Grad-CAM uses the same predicted class as the existing inference pipeline. If the explanation graph cannot be verified against the prediction, the explanation is withheld rather than displaying a potentially misleading visualization. The explanation model is cached in process; it does not replace or save a model. The PNG overlay is generated in memory and is not persisted.

Grad-CAM provides a qualitative explanation of image regions that influenced the model prediction. It is not a validated disease segmentation map or crop-loss measurement. It does not establish exact lesion boundaries, disease severity, insurance eligibility, or that the highlighted area contains a disease. For healthy predictions it explains regions influencing the healthy-class prediction; it does not identify damage. The visualization has no pixel-level localization ground truth and its scientific localization accuracy has not been validated.

## Limits

Controlled images can produce optimistic within-dataset scores and poor field generalization. A confidence value is a model score, not a probability of a correct diagnosis unless separately calibrated. Novel crops, multiple leaves, poor lighting, non-leaf damage, and field backgrounds may lead to errors.
