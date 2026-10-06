"""Train a MobileNetV2 crop/disease classifier from prepared folders."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ml.config import IMAGE_SIZE, RESULTS_DIR, SEED, SPLIT_DIRS, MODEL_DIR
from ml.preprocess import write_preprocessing_config


def train(epochs=12, batch_size=24, seed=SEED):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = tf.keras.utils.image_dataset_from_directory(
            SPLIT_DIRS[split], image_size=IMAGE_SIZE, batch_size=batch_size,
            label_mode="categorical", shuffle=split == "train", seed=seed)
    labels = datasets["train"].class_names
    if labels != datasets["validation"].class_names:
        raise ValueError("Train and validation class directories differ")
    augment = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomZoom(0.1)])
    base = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(labels), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
                 tf.keras.callbacks.ModelCheckpoint(MODEL_DIR / "crop_disease.keras", save_best_only=True, monitor="val_loss")]
    history = model.fit(datasets["train"].prefetch(tf.data.AUTOTUNE), validation_data=datasets["validation"].prefetch(tf.data.AUTOTUNE), epochs=epochs, callbacks=callbacks)
    write_preprocessing_config(MODEL_DIR / "model_config.json", labels)
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    return model, labels, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args(); train(args.epochs, args.batch_size)
