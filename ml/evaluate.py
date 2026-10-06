"""Evaluate a saved model on the untouched test split."""
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from ml.config import IMAGE_SIZE, MODEL_DIR, RESULTS_DIR, SPLIT_DIRS


def evaluate():
    model = tf.keras.models.load_model(MODEL_DIR / "crop_disease.keras")
    ds = tf.keras.utils.image_dataset_from_directory(SPLIT_DIRS["test"], image_size=IMAGE_SIZE, batch_size=32, shuffle=False)
    labels = ds.class_names
    truth = np.concatenate([y.numpy() for _, y in ds])
    probabilities = model.predict(ds, verbose=0)
    pred = probabilities.argmax(axis=1)
    report = classification_report(truth, pred, target_names=labels, output_dict=True, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(truth, pred)), "classification_report": report,
               "test_examples": int(len(truth)), "classes": labels}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for label in labels:
            values = report[label]
            writer.writerow([label, values["precision"], values["recall"], values["f1-score"], values["support"]])
        writer.writerow(["accuracy", metrics["accuracy"], "", "", len(truth)])
    split_summary = SPLIT_DIRS["train"].parent / "split_summary.json"
    split_info = json.loads(split_summary.read_text(encoding="utf-8")) if split_summary.exists() else {}
    report_lines = ["# Model evaluation report", "", "Metrics from the held-out test split for this trained artifact.", "",
                    f"- Test examples: {len(truth)}", f"- Classes: {len(labels)}", f"- Accuracy: {metrics['accuracy']:.4f}",
                    f"- Macro precision: {report['macro avg']['precision']:.4f}",
                    f"- Macro recall: {report['macro avg']['recall']:.4f}", f"- Macro F1: {report['macro avg']['f1-score']:.4f}",
                    f"- Training seed: {split_info.get('seed', 'not recorded')}", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in labels:
        values = report[label]
        report_lines.append(f"| {label} | {values['precision']:.3f} | {values['recall']:.3f} | {values['f1-score']:.3f} | {int(values['support'])} |")
    (RESULTS_DIR / "evaluation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    cm = confusion_matrix(truth, pred)
    np.savetxt(RESULTS_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(16, 14)); ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation="vertical", colorbar=False)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=140); plt.close(fig)
    history_path = RESULTS_DIR / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, key, title in zip(axes, ("loss", "accuracy"), ("Training loss", "Training accuracy")):
            if key in history:
                axis.plot(history[key], label="train")
            if f"val_{key}" in history:
                axis.plot(history[f"val_{key}"], label="validation")
            axis.set_title(title); axis.set_xlabel("Epoch"); axis.legend()
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "training_curves.png", dpi=140); plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
