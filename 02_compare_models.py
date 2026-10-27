"""
02_compare_models.py
---------------------
OPTIONAL but recommended: run this AFTER 01_inspect_dataset.py and BEFORE
the full training run in 03_train.py.

It trains each candidate backbone for a small number of "probe" epochs
(frozen base, same data/augmentation you'll use for real training) and
records:
  - validation accuracy after the probe epochs
  - average seconds/epoch on YOUR GPU
  - parameter count and .h5 size on disk
  - a simple composite score

This gives you real, dataset-specific evidence for which backbone to commit
to for the full 10-15 epoch run, instead of picking one from memory.

Usage:
    python 02_compare_models.py --data_dir "D:/path/to/dataset" --probe_epochs 3

Output:
    model_comparison.json / model_comparison.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

CANDIDATES = {
    "MobileNetV2": dict(
        app=tf.keras.applications.MobileNetV2,
        preprocess=tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "EfficientNetB0": dict(
        app=tf.keras.applications.EfficientNetB0,
        preprocess=tf.keras.applications.efficientnet.preprocess_input,
    ),
    "DenseNet121": dict(
        app=tf.keras.applications.DenseNet121,
        preprocess=tf.keras.applications.densenet.preprocess_input,
    ),
    "ResNet50": dict(
        app=tf.keras.applications.ResNet50,
        preprocess=tf.keras.applications.resnet50.preprocess_input,
    ),
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_datasets(data_dir, preprocess_fn, seed=42):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)

    augment = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ])

    def prep_train(x, y):
        x = augment(x, training=True)
        x = preprocess_fn(x)
        return x, y

    def prep_val(x, y):
        x = preprocess_fn(x)
        return x, y

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_val, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names, num_classes


def build_model(app_fn, num_classes):
    base = app_fn(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--probe_epochs", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=list(CANDIDATES.keys()))
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible to TensorFlow: {gpus}")
    if not gpus:
        print("WARNING: no GPU detected by TensorFlow — this probe (and full "
              "training) will run on CPU and be slow. Check your CUDA/cuDNN "
              "install if you have an NVIDIA GPU.")

    results = []
    for name in args.models:
        cfg = CANDIDATES[name]
        print(f"\n=== Probing {name} ===")
        train_ds, val_ds, class_names, num_classes = build_datasets(args.data_dir, cfg["preprocess"])
        model = build_model(cfg["app"], num_classes)

        n_params = model.count_params()

        epoch_times = []
        class TimeCB(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self._t0 = time.time()
            def on_epoch_end(self, epoch, logs=None):
                epoch_times.append(time.time() - self._t0)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.probe_epochs,
            callbacks=[TimeCB()],
            verbose=1,
        )

        tmp_path = Path(f"_probe_{name}.h5")
        model.save(tmp_path)
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        tmp_path.unlink()

        val_acc = history.history["val_accuracy"][-1]
        avg_epoch_s = sum(epoch_times) / len(epoch_times)

        results.append({
            "model": name,
            "probe_val_accuracy": round(float(val_acc), 4),
            "avg_seconds_per_epoch": round(avg_epoch_s, 1),
            "params_millions": round(n_params / 1e6, 2),
            "h5_size_mb": round(size_mb, 1),
        })
        print(f"{name}: val_acc={val_acc:.4f}  s/epoch={avg_epoch_s:.1f}  "
              f"params={n_params/1e6:.2f}M  size={size_mb:.1f}MB")

        tf.keras.backend.clear_session()

    # simple composite score: prioritize accuracy, penalize slow/large models
    max_acc = max(r["probe_val_accuracy"] for r in results) or 1e-9
    max_time = max(r["avg_seconds_per_epoch"] for r in results) or 1e-9
    max_size = max(r["h5_size_mb"] for r in results) or 1e-9
    for r in results:
        r["score"] = round(
            0.6 * (r["probe_val_accuracy"] / max_acc)
            + 0.25 * (1 - r["avg_seconds_per_epoch"] / max_time)
            + 0.15 * (1 - r["h5_size_mb"] / max_size),
            4,
        )
    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n=== Comparison (sorted by composite score) ===")
    for r in results:
        print(r)

    with open("model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("model_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRecommended backbone based on THIS probe run: {results[0]['model']}")
    print("This is based on a short probe, not the full training run — treat it "
          "as evidence, re-check after the real training in 03_train.py, and "
          "note that real accuracy will differ from this short-probe number.")


if __name__ == "__main__":
    main()
