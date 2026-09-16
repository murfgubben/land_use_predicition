"""EuroSAT indexing, stratified splitting, and tf.data input pipelines."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import tifffile
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import DATA_CONFIG, DATA_DIR


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    """A single image path and its integer class label."""

    path: str
    label: int


@dataclass(frozen=True)
class DatasetSplits:
    """The three disjoint data splits and their class mapping."""

    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    class_names: tuple[str, ...]
    train_records: tuple[ImageRecord, ...]
    validation_records: tuple[ImageRecord, ...]
    test_records: tuple[ImageRecord, ...]


def _find_class_root(data_dir: Path) -> Path:
    """Find a directory whose immediate children are class directories."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}. "
            "Set SATELLITE_DATA_DIR or place EuroSAT under data/."
        )

    candidates = [data_dir]
    candidates.extend(path for path in data_dir.iterdir() if path.is_dir())
    for candidate in candidates:
        class_directories = [
            path for path in candidate.iterdir() if path.is_dir()
        ]
        if len(class_directories) >= 2 and all(
            any(
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
                for image_path in class_directory.iterdir()
            )
            for class_directory in class_directories
        ):
            return candidate

    raise ValueError(
        f"Could not find class directories containing images under {data_dir}. "
        "Expected a layout such as data/<class_name>/<image>."
    )


def index_images(
    data_dir: Path = DATA_DIR,
) -> tuple[list[ImageRecord], tuple[str, ...]]:
    """Index images in a class-directory dataset using a stable class mapping."""
    class_root = _find_class_root(data_dir)
    class_directories = sorted(
        path for path in class_root.iterdir() if path.is_dir()
    )
    class_names = tuple(path.name for path in class_directories)
    class_to_index = {name: index for index, name in enumerate(class_names)}

    records: list[ImageRecord] = []
    for class_directory in class_directories:
        image_paths = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        records.extend(
            ImageRecord(str(path), class_to_index[class_directory.name])
            for path in image_paths
        )

    if not records:
        raise ValueError(f"No supported image files found under {class_root}.")

    return records, class_names


def _validate_split_ratios() -> None:
    ratios = (
        DATA_CONFIG.train_ratio,
        DATA_CONFIG.validation_ratio,
        DATA_CONFIG.test_ratio,
    )
    if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise ValueError(f"Split ratios must be between 0 and 1: {ratios}")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0: {ratios}")


def split_records(
    records: list[ImageRecord],
) -> tuple[
    tuple[ImageRecord, ...],
    tuple[ImageRecord, ...],
    tuple[ImageRecord, ...],
]:
    """Create deterministic, stratified train/validation/test records."""
    _validate_split_ratios()
    labels = [record.label for record in records]
    train_records, temporary_records = train_test_split(
        records,
        test_size=1.0 - DATA_CONFIG.train_ratio,
        random_state=DATA_CONFIG.seed,
        stratify=labels,
    )
    temporary_labels = [record.label for record in temporary_records]
    test_fraction_of_temporary = DATA_CONFIG.test_ratio / (
        DATA_CONFIG.validation_ratio + DATA_CONFIG.test_ratio
    )
    validation_records, test_records = train_test_split(
        temporary_records,
        test_size=test_fraction_of_temporary,
        random_state=DATA_CONFIG.seed,
        stratify=temporary_labels,
    )
    return tuple(train_records), tuple(validation_records), tuple(test_records)


def _read_tiff(contents: np.ndarray) -> np.ndarray:
    """Read an all-bands TIFF and return its RGB bands as float32."""
    image = tifffile.imread(BytesIO(contents.item()))
    if image.ndim != 3:
        raise ValueError(
            f"Expected a multi-band TIFF with shape [height, width, bands], "
            f"got {image.shape}."
        )
    if max(DATA_CONFIG.tiff_rgb_bands) >= image.shape[-1]:
        raise ValueError(
            f"TIFF has {image.shape[-1]} bands, but RGB band indexes are "
            f"{DATA_CONFIG.tiff_rgb_bands}."
        )
    rgb = image[..., DATA_CONFIG.tiff_rgb_bands].astype(np.float32)
    return np.clip(
        rgb / DATA_CONFIG.tiff_reflectance_scale * 255.0, 0.0, 255.0
    )


def _decode_tiff(path: tf.Tensor) -> tf.Tensor:
    contents = tf.io.read_file(path)
    image = tf.numpy_function(_read_tiff, [contents], tf.float32)
    image.set_shape([None, None, len(DATA_CONFIG.tiff_rgb_bands)])
    return image


def _decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    is_tiff = tf.strings.regex_full_match(
        tf.strings.lower(path), r".*\.(tif|tiff)"
    )
    image = tf.cond(
        is_tiff,
        lambda: _decode_tiff(path),
        lambda: tf.image.decode_image(
            image, channels=3, expand_animations=False
        ),
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, DATA_CONFIG.image_size)
    return tf.cast(image, tf.float32), label


def _augment_image(
    image: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image, seed=DATA_CONFIG.seed)
    image = tf.image.random_flip_up_down(image, seed=DATA_CONFIG.seed + 1)
    rotation_count = tf.random.uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32, seed=DATA_CONFIG.seed
    )
    image = tf.image.rot90(image, k=rotation_count)
    return image, label


def _make_dataset(
    records: tuple[ImageRecord, ...], training: bool
) -> tf.data.Dataset:
    paths = [record.path for record in records]
    labels = [record.label for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            min(len(records), DATA_CONFIG.shuffle_buffer_size),
            seed=DATA_CONFIG.seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        _decode_image, num_parallel_calls=DATA_CONFIG.num_parallel_calls
    )
    if training:
        dataset = dataset.map(
            _augment_image, num_parallel_calls=DATA_CONFIG.num_parallel_calls
        )
    return dataset.batch(DATA_CONFIG.batch_size).prefetch(tf.data.AUTOTUNE)


def load_dataset_splits(data_dir: Path = DATA_DIR) -> DatasetSplits:
    """Index EuroSAT, split it once, and build the three tf.data datasets."""
    records, class_names = index_images(data_dir)
    train_records, validation_records, test_records = split_records(records)
    return DatasetSplits(
        train=_make_dataset(train_records, training=True),
        validation=_make_dataset(validation_records, training=False),
        test=_make_dataset(test_records, training=False),
        class_names=class_names,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
    )
