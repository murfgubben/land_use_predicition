"""Central configuration for the satellite image classifier."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SATELLITE_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = Path(
    os.getenv("SATELLITE_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
)


@dataclass(frozen=True)
class DataConfig:
    """Dataset and input-pipeline settings."""

    image_size: tuple[int, int] = (64, 64)
    # The classifier is intentionally RGB-only. For TIFF inputs, we discard the
    # remaining spectral bands and keep only B4/B3/B2 to match standard RGB.
    input_channels: int = 3
    # EuroSAT all-bands TIFFs use Sentinel-2 band order:
    # B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12.
    # Select B4/B3/B2 to provide the RGB input expected by the classifier.
    tiff_rgb_bands: tuple[int, int, int] = (3, 2, 1)
    tiff_reflectance_scale: float = 10_000.0
    batch_size: int = 32
    seed: int = 42
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle_buffer_size: int = 2_048
    num_parallel_calls: int = -1


@dataclass(frozen=True)
class ModelConfig:
    """Model settings used by the transfer-learning model."""

    backbone_name: str = "MobileNetV2"
    backbone_weights: str | None = "imagenet"
    dropout_rate: float = 0.2
    initial_learning_rate: float = 1e-3
    fine_tuning_learning_rate: float = 1e-5
    fine_tune_layers: int = 20
    artifact_filename: str = "satellite_classifier.keras"
    class_mapping_filename: str = "class_mapping.json"
    metrics_filename: str = "training_metrics.csv"


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop settings."""

    epochs: int = 15
    fine_tune_epochs: int = 5
    verbose: int = 1


DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
