# Satellite classifier

Training code for a CPU-friendly EuroSAT image classifier.

## Dataset layout

Place the downloaded EuroSAT RGB dataset under `data/`, with one directory per
class:

```text
data/
└── AnnualCrop/
    ├── image_1.jpg
    └── ...
```

The loader also accepts one wrapper directory, such as
`data/EuroSAT_RGB/AnnualCrop/image_1.jpg`. Set `SATELLITE_DATA_DIR` to use a
different location.

The all-bands EuroSAT distribution is supported as well:

```text
data/
└── EuroSATallBands/
    └── AnnualCrop/
        └── AnnualCrop_1.tif
```

These TIFFs contain 13 Sentinel-2 bands. The loader selects B4/B3/B2 (true
color) so the input remains compatible with the classifier's three-channel
backbone. The accompanying CSV split files and `label_map.json` are not
required; class directories are used as the source of truth.

## Current pipeline

`src/data.py` creates a deterministic, stratified 70/15/15 train/validation/test
split. Class names are sorted before assigning integer labels, which keeps the
mapping stable for the eventual serving component. Training data receives only
horizontal and vertical flips plus random rotations by 0, 90, 180, or 270
degrees; validation and test data are not augmented. The split seed and all
pipeline settings are centralized in `src/config.py`.

Install the pinned dependencies with:

```bash
python3 -m pip install -r requirements.txt
```
