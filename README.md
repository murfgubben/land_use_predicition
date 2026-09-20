# Satellite image classifier

This repository trains an image classifier for the [EuroSAT](https://github.com/phelber/EuroSAT)
land-use and land-cover dataset. It is a small, CPU-friendly TensorFlow
project, but the same structure is useful for understanding a typical
supervised image-classification workflow:

1. discover and label images from a directory tree;
2. create deterministic training, validation, and test splits;
3. decode and normalize images with a `tf.data` input pipeline;
4. train a transfer-learning model;
5. save the model and its class-label mapping; and
6. evaluate once on held-out test data.

The implementation lives in the [`src/`](./src/) package. Configuration is
centralized in [`src/config.py`](./src/config.py), so the commands below work
without editing source files for ordinary runs.

## Quick start

Run these commands from the repository root.

### 1. Create an environment

Using a virtual environment keeps the pinned dependencies isolated from other
Python projects:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The dependency versions are pinned in [`requirements.txt`](./requirements.txt).
TensorFlow can use a CPU-only installation; a GPU is not required.

### 2. Put the dataset in place

The default location is `data/`. See [Dataset layout](#dataset-layout) for the
supported directory structures.

### 3. Train

```bash
python3 -m src.train
```

Training creates an `artifacts/` directory containing the saved model, the
class mapping, and a CSV log of training metrics. The first run may download
MobileNetV2's ImageNet weights, so network access is needed unless those
weights are already cached.

### 4. Evaluate

```bash
python3 -m src.evaluate
```

Evaluation reloads the saved model, rebuilds the same deterministic split, and
reports test accuracy, per-class precision/recall/F1, and a confusion matrix.
The test set is not used during training.

## Dataset layout

The loader searches the supplied data directory and one level of wrapper
directories for class directories. Each class directory must contain at least
one supported image file. Class-directory names become the class names, and
are sorted alphabetically before integer labels are assigned.

### EuroSAT RGB

```text
data/
├── AnnualCrop/
│   ├── image_1.jpg
│   └── ...
├── Forest/
│   └── ...
└── ...
```

A wrapper directory is also accepted:

```text
data/
└── EuroSAT_RGB/
    ├── AnnualCrop/
    └── Forest/
```

Supported RGB file extensions are `.bmp`, `.jpeg`, `.jpg`, and `.png`.

### EuroSAT all-bands TIFF

```text
data/
└── EuroSATallBands/
    ├── AnnualCrop/
    │   └── AnnualCrop_1.tif
    └── Forest/
        └── Forest_1.tif
```

The all-bands EuroSAT files contain Sentinel-2 bands in this order:

```text
B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
```

The classifier expects three channels, so the loader selects B4, B3, and B2
(true color). TIFF reflectance values are divided by `10,000`, clipped to the
range `0..1`, and converted to the equivalent `0..255` image range before
resizing. The accompanying CSV split files and `label_map.json` are not
required; the class directories are the source of truth.

## How the code works

### Indexing and splitting

[`src/data.py`](./src/data.py) indexes every supported image and assigns a
zero-based integer label based on the sorted class names. It then makes a
stratified, deterministic split:

| Split | Proportion | Purpose |
| --- | ---: | --- |
| Training | 70% | Fit model parameters |
| Validation | 15% | Monitor performance during training |
| Test | 15% | Final, held-out evaluation |

The random seed is `42` by default. Stratification preserves approximately the
same class proportions in each split. Because the split is reconstructed by
both training and evaluation, use the same dataset contents and configuration
when comparing results. Adding, removing, or renaming images changes the
split.

### Input pipeline

The training and evaluation datasets are TensorFlow `tf.data.Dataset` objects:

1. files are optionally shuffled for training;
2. RGB images or TIFFs are decoded;
3. images are resized to `64 x 64`;
4. images are represented as `float32`;
5. training images receive random horizontal and vertical flips and a random
   rotation by 0, 90, 180, or 270 degrees;
6. images are batched in groups of 32 and prefetched.

Validation and test images are resized but not augmented. Augmentation is
applied only to training data so evaluation measures the model on unchanged
examples.

### Model and training phases

[`src/model.py`](./src/model.py) builds a MobileNetV2 transfer-learning model:

```text
64x64 RGB image
    -> MobileNetV2 convolutional backbone (ImageNet weights)
    -> global average pooling
    -> dropout
    -> softmax class probabilities
```

MobileNetV2's ImageNet preprocessing is applied inside the model. The
convolutional backbone is initially frozen, and only the new classification
layer is trained for 15 epochs with Adam at a learning rate of `1e-3`.

If `fine_tune_epochs` is greater than zero, training continues for five more
epochs. The final 20 backbone layers are unfrozen, batch-normalization layers
remain frozen, and the learning rate is reduced to `1e-5`. This second phase
adapts high-level ImageNet features to satellite imagery without making large
updates that could destroy useful pretrained features.

### Artifacts

The default `artifacts/` directory contains:

| File | Description |
| --- | --- |
| `satellite_classifier.keras` | Complete Keras model, including architecture and weights |
| `class_mapping.json` | Mapping from output index (`"0"`, `"1"`, ...) to class name |
| `training_metrics.csv` | Epoch-by-epoch loss and accuracy written by Keras |

The class mapping is part of the model output contract. Evaluation checks that
it still matches the class ordering discovered in the dataset before producing
metrics.

## Commands and paths

Both entry points accept explicit path overrides:

```bash
python3 -m src.train \
  --data-dir /path/to/dataset \
  --artifacts-dir /path/to/output

python3 -m src.evaluate \
  --data-dir /path/to/dataset \
  --artifacts-dir /path/to/output
```

You can also set defaults through environment variables:

```bash
export SATELLITE_DATA_DIR=/path/to/dataset
export SATELLITE_ARTIFACTS_DIR=/path/to/output
python3 -m src.train
```

Command-line arguments take precedence over the environment-backed defaults.

## Configuration

Edit the frozen dataclasses in [`src/config.py`](./src/config.py) when changing
the experiment itself:

- `DataConfig`: image size, batch size, seed, split proportions, TIFF band
  indexes, and input-pipeline parallelism;
- `ModelConfig`: MobileNetV2 weights, dropout, learning rates, fine-tuning
  depth, and artifact filenames;
- `TrainingConfig`: initial epochs, fine-tuning epochs, and console verbosity.

If you change the image size, split ratios, class mapping rules, or model
configuration, retrain and evaluate a new artifact set. Do not mix an old
`class_mapping.json` or model with a differently configured dataset.

## Reading the evaluation output

The evaluation command prints:

- **Test accuracy**: the fraction of test images whose predicted class is
  correct;
- **Precision**: among images predicted as a class, how many really belong to
  it;
- **Recall**: among images belonging to a class, how many were found;
- **F1-score**: the harmonic mean of precision and recall; and
- **Confusion matrix**: rows are true classes and columns are predicted
  classes, in the order shown by `class_mapping.json`.

Accuracy can hide poor performance on an individual class. Inspect the
per-class report and confusion matrix, especially when classes have different
visual difficulty or when the dataset is not perfectly balanced.

## Troubleshooting

### CUDA, GPU, or TensorRT messages

Messages such as `Could not find cuda drivers`, `CUDA_ERROR_NO_DEVICE`, or
`Could not find TensorRT` are informational warnings on a CPU-only machine.
Training still runs on the CPU. They are not errors unless you explicitly
expect GPU acceleration.

### Dataset not found

Check that the path exists and contains class directories directly or beneath
one wrapper directory:

```bash
find data -maxdepth 2 -type d
```

Use `--data-dir` or `SATELLITE_DATA_DIR` if the dataset is stored elsewhere.

### Saved model or mapping not found

Run training first, or point evaluation at the directory produced by training:

```bash
python3 -m src.evaluate --artifacts-dir /path/to/training/artifacts
```

### Predict one image

After training, run inference on any supported image size or aspect ratio.
JPEG (`.jpg`/`.jpeg`) and PNG (`.png`) files are supported, including
grayscale and RGBA images:

```bash
python3 -m src.predict path/to/image.jpg
# Direct script execution is also supported:
python3 src/predict.py path/to/image.png
```

The prediction loader converts images to RGB, resizes them to `64 x 64`, and
uses the same input preprocessing as the training pipeline. It prints all
class probabilities from most to least likely. The default model and class
mapping are read from `artifacts/`; set `SATELLITE_ARTIFACTS_DIR` to use a
different artifacts directory. The CLI also saves polished visualizations to
`outputs/comparison.png` and `outputs/confidence.png`; set
`SATELLITE_OUTPUTS_DIR` to use a different output directory.

### Local FastAPI service

Install the dependencies, then start the service from the project root:

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn src.api:app --reload
```

Check liveness:

```bash
curl http://127.0.0.1:8000/health
```

Send a sample JPEG or PNG to `/predict`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/sample.jpg"
```

The visualization endpoint returns a PNG directly:

```bash
curl -X POST http://127.0.0.1:8000/predict/visualize \
  -F "file=@path/to/sample.jpg" \
  --output confidence.png
```

Uploads are limited to 10 MB and support JPEG, PNG, and TIFF files. The
trained model and class mapping are loaded once during application startup.

### Class mapping mismatch

This means the saved model was produced from a different class ordering or
dataset than the one currently being evaluated. Keep the original dataset
unchanged, or remove the old artifacts and train a new model.

### TIFF errors

All-bands TIFF files must contain the configured Sentinel-2 bands. The current
configuration selects zero-based band indexes `(3, 2, 1)`, corresponding to
B4/B3/B2. Ordinary three-channel RGB images should use JPG, JPEG, PNG, or BMP
files instead of being placed in the TIFF pipeline.

## Project structure

```text
.
├── src/
│   ├── config.py    # Paths and experiment settings
│   ├── data.py      # Indexing, splitting, decoding, augmentation
│   ├── model.py     # MobileNetV2 model and fine-tuning
│   ├── train.py     # Training entry point and artifact writing
│   ├── evaluate.py  # Held-out test evaluation
│   ├── predict.py   # Prediction and CLI inference
│   ├── visualize.py # Prediction visualization helpers
│   └── api.py       # FastAPI service
├── requirements.txt # Pinned Python dependencies
├── data/             # Local dataset; ignored by Git
└── artifacts/        # Generated outputs; ignored by Git
```

Generated datasets, model files, Python bytecode, and virtual environments are
ignored by Git. Keep large datasets and trained artifacts outside version
control unless a separate artifact-storage workflow is introduced.
