# Skin Disease Detection — local GPU training pipeline

Everything here runs on your machine, on your NVIDIA GPU. No Colab, no Drive.

## 0. Before you start: GPU + Windows note

TensorFlow **dropped native Windows GPU support after version 2.10** — TF
2.11+ only gets GPU acceleration on Windows through **WSL2** (Linux
subsystem) or Linux directly.

- If you're on native Windows (not WSL2): `pip install "tensorflow==2.10.1"`
  and install CUDA 11.2 / cuDNN 8.1 to match it, **not** the
  `tensorflow[and-cuda]` line in requirements.txt.
- If you're on WSL2 or Linux: `tensorflow[and-cuda]==2.15.0` (as in
  requirements.txt) will pull in matching CUDA/cuDNN automatically.

Either way, verify TensorFlow actually sees your GPU before training:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If that prints an empty list, fix the CUDA/cuDNN/driver setup first —
`03_train.py` will still run on CPU but will be far slower, and it will
print a warning either way.

## 1. Set up the environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 2. Download and extract the dataset

Download `skin-diseases-image-dataset` from Kaggle and extract it locally,
e.g. to `D:\datasets\skin-diseases-image-dataset`. This dataset ships as
**10 classes**, one folder per disease — but step 3 auto-detects the real
layout on your disk rather than assuming that, in case Kaggle's packaging
changes.

## 3. Inspect the dataset (classes, counts, corrupted images)

```bash
python 01_inspect_dataset.py --data_dir "D:\datasets\skin-diseases-image-dataset"
```
This writes `dataset_report.json` (used by later steps) and
`corrupted_files.txt`. Read the console output — check the class list and
per-class counts look right, and check whether any images were flagged as
corrupted before you continue. Corrupted files are **not deleted
automatically**; `image_dataset_from_directory` in the next steps will
simply fail to decode them if you leave them in place, so delete anything
listed in `corrupted_files.txt` if the count is non-zero.

## 4. (Recommended) Compare candidate backbones on your data

```bash
python 02_compare_models.py --data_dir "D:\datasets\skin-diseases-image-dataset" --probe_epochs 3
```
This trains MobileNetV2, EfficientNetB0, DenseNet121, and ResNet50 for a
few frozen-base epochs each, on your actual data, and reports validation
accuracy, seconds/epoch on your GPU, parameter count, and `.h5` size in
`model_comparison.json/csv`. Use its top pick (or your own judgment on the
accuracy/speed/size trade-off it shows) as `--model_name` in step 5.

**Why EfficientNetB0 is the practical default if you skip this step:**
for a moderate-size, moderately-imbalanced medical image dataset like this
one, EfficientNetB0 typically lands in the same accuracy range as
ResNet50/DenseNet121 on ImageNet-pretrained transfer learning, but with
~4-5x fewer parameters (~5.3M vs ~23-25M) and a much smaller `.h5` file,
which means faster epochs and lower GPU memory pressure — useful if your
GPU has limited VRAM. MobileNetV2 is smaller and faster still but usually
trades a bit of accuracy for that speed. ResNet50/DenseNet121 can win on
raw accuracy with enough data and fine-tuning, at the cost of longer
training time and a larger model file. **This is a general prior, not a
promise about your dataset** — step 4's comparison is what should actually
decide it, and the "why this model" explanation you get at the end of
training is based on the real numbers from your run, not this default.

## 5. Train

```bash
python 03_train.py --data_dir "D:\datasets\skin-diseases-image-dataset" --model_name EfficientNetB0
```
Key behavior:
- 80/20 train/val split via `image_dataset_from_directory` (deterministic,
  seeded).
- Resize to 224×224, RGB.
- Normalization: whichever `preprocess_input` matches `--model_name`
  (e.g. EfficientNet's is a rescale, ResNet50/DenseNet121's is Caffe-style
  channel normalization, MobileNetV2's is `[-1, 1]` scaling) — this matters
  for pretrained-weight accuracy and is applied automatically.
- Augmentation (train split only): horizontal flip, small rotation, zoom,
  width/height shifts.
- Phase 1: frozen base, GlobalAveragePooling2D → Dense(128, relu) →
  Dropout(0.3) → Dense(softmax), Adam, categorical cross-entropy, up to
  `--epochs` (default 15) with EarlyStopping/ModelCheckpoint/ReduceLROnPlateau.
- Phase 2 (default on, disable with `--no_finetune`): unfreezes the top
  layers of the base and fine-tunes at `1e-5` for `--finetune_epochs`
  (default 8).
- To also score a folder of images the model never saw during training:
  add `--test_dir "D:\some\unseen_images"`.

Outputs (in the working directory, or `--out_dir` if you set one):
- `skin_model_kaggle.h5`
- `class_names.json` — exact `{index: class_name}` mapping matching the
  model's output layer, for `app.py` to reuse.
- `training_curves.png` — accuracy/loss plots.
- `confusion_matrix.png`, `classification_report.txt` — on the validation
  split.
- `unseen_predictions.json` — if `--test_dir` was given.
- `run_summary.json` — the **actual** final train/val accuracy and loss
  from this run. Nothing here is invented or pre-guaranteed; re-running
  will shift these numbers slightly because of augmentation randomness.

## 6. Launch the Gradio app

```bash
python app.py
```
Upload an image; it's resized to 224×224, RGB, run through the same
`preprocess_input` as training, and the app prints **only the predicted
class name** (no confidence number), by design.

If you trained with a backbone other than EfficientNetB0, edit the
`preprocess_input` import at the top of `app.py` to match — the mapping is
listed right there in a comment.

## Files in this folder

| File | Purpose |
|---|---|
| `01_inspect_dataset.py` | Auto-detects classes, counts, corrupted images |
| `02_compare_models.py` | Optional backbone bake-off on your data/GPU |
| `03_train.py` | Full training pipeline, evaluation, artifact export |
| `app.py` | Gradio inference app |
| `requirements.txt` | Pinned dependencies |

## Honesty note

I have not run this pipeline myself — I don't have access to your GPU or
the extracted dataset in this environment, so I can't hand you real
accuracy numbers or a pre-picked "winning" model. `run_summary.json` from
your own run in step 5 (and `model_comparison.json` from step 4) are the
real numbers, produced on your data and your hardware — use those, not any
number quoted in general ML literature, to decide whether the model is
good enough for your purposes.
