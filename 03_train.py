"""
03_train.py  (CPU run, aiming for higher accuracy)
------------------------------------------------------
EfficientNetB0, 19,183 images, 9 classes, 224x224.
Adds back vs. the fast-CPU variant:
  - fine-tuning phase (unfreezes top EfficientNetB0 blocks, low LR)
  - class_weight to counter the imbalance (Melanoma ~15.75k vs
    Atopic Dermatitis ~1.25k) that was hurting minority-class recall
  - ReduceLROnPlateau, since we now train enough epochs for it to matter
No accuracy is guaranteed -- these are the levers most likely to help,
not a promise of 80%.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
SEED = 42

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

EFFNET = dict(
    app=tf.keras.applications.EfficientNetB0,
    preprocess=tf.keras.applications.efficientnet.preprocess_input,
    finetune_from_ratio=0.8,  # unfreeze top ~20% of base layers
)


def build_datasets(data_dir, preprocess_fn, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical",
    )
    class_names = train_ds.class_names

    # compute class weights from the training split's true label distribution
    train_labels = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in train_ds])
    class_weight_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=train_labels,
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weight_arr)}
    print("Class weights (balanced):", {class_names[i]: round(w, 2) for i, w in class_weight.items()})

    augment = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

    def prep_train(x, y):
        x = augment(x, training=True)
        x = preprocess_fn(x)
        return x, y

    def prep_val(x, y):
        x = preprocess_fn(x)
        return x, y

    train_ds_final = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    val_ds_final = val_ds.map(prep_val, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds_final, val_ds_final, class_names, class_weight


def build_model(app_fn, num_classes, dropout=0.3, dense_units=128):
    base = app_fn(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def plot_history(histories, out_path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, acc, label="train"); axes[0].plot(epochs, val_acc, label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, loss, label="train"); axes[1].plot(epochs, val_loss, label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved training curves -> {out_path}")


def evaluate_and_report(model, val_ds, class_names, out_dir):
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred += list(np.argmax(preds, axis=1))
        y_true += list(np.argmax(y_batch.numpy(), axis=1))

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print("\n=== Classification report (validation split) ===")
    print(report)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6), max(5, len(class_names) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (validation split)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)
    print(f"Saved confusion_matrix.png and classification_report.txt -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8, help="frozen-head phase")
    ap.add_argument("--finetune_epochs", type=int, default=6)
    ap.add_argument("--no_finetune", action="store_true")
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPUs visible to TensorFlow: {tf.config.list_physical_devices('GPU')}")

    cfg = EFFNET
    train_ds, val_ds, class_names, class_weight = build_datasets(args.data_dir, cfg["preprocess"], args.batch_size)
    num_classes = len(class_names)
    print(f"Classes detected ({num_classes}): {class_names}")

    model, base = build_model(cfg["app"], num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    ckpt_path = str(out_dir / "skin_model_kaggle.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    fit_kwargs = {} if args.no_class_weight else {"class_weight": class_weight}

    print(f"\n=== Phase 1: training head, base frozen ({args.epochs} epochs max) ===")
    t0 = time.time()
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                          callbacks=callbacks, **fit_kwargs)
    print(f"Phase 1 wall time: {time.time() - t0:.1f}s")
    histories = [history1]

    if not args.no_finetune:
        print("\n=== Phase 2: fine-tuning top EfficientNetB0 blocks ===")
        base.trainable = True
        n_layers = len(base.layers)
        freeze_up_to = int(n_layers * cfg["finetune_from_ratio"])
        for layer in base.layers[:freeze_up_to]:
            layer.trainable = False
        print(f"Unfroze {n_layers - freeze_up_to}/{n_layers} top base layers.")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        history2 = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                              callbacks=callbacks, **fit_kwargs)
        print(f"Phase 2 wall time: {time.time() - t0:.1f}s")
        histories.append(history2)

    final_train_acc = histories[-1].history["accuracy"][-1]
    final_val_acc = histories[-1].history["val_accuracy"][-1]
    final_train_loss = histories[-1].history["loss"][-1]
    final_val_loss = histories[-1].history["val_loss"][-1]
    print("\n=== Final metrics from THIS run ===")
    print(f"train_accuracy={final_train_acc:.4f}  val_accuracy={final_val_acc:.4f}")
    print(f"train_loss={final_train_loss:.4f}      val_loss={final_val_loss:.4f}")

    plot_history(histories, out_dir / "training_curves.png")
    evaluate_and_report(model, val_ds, class_names, out_dir)

    model.save(ckpt_path)
    class_map = {str(i): name for i, name in enumerate(class_names)}
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_map, f, indent=2)

    run_summary = {
        "model_name": "EfficientNetB0",
        "num_classes": num_classes,
        "class_names": class_names,
        "final_train_accuracy": round(float(final_train_acc), 4),
        "final_val_accuracy": round(float(final_val_acc), 4),
        "final_train_loss": round(float(final_train_loss), 4),
        "final_val_loss": round(float(final_val_loss), 4),
        "finetuned": not args.no_finetune,
        "class_weighted": not args.no_class_weight,
        "epochs_requested": args.epochs,
        "finetune_epochs_requested": args.finetune_epochs,
        "batch_size": args.batch_size,
        
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"\nSaved: {ckpt_path}")
    print(f"Saved: {out_dir / 'class_names.json'}")
    print(f"Saved: {out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()\
        