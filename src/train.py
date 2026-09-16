"""Command-line training entry point."""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from .config import (
    ARTIFACTS_DIR,
    DATA_CONFIG,
    DATA_DIR,
    MODEL_CONFIG,
    TRAINING_CONFIG,
)
from .data import load_dataset_splits
from .model import create_model, enable_fine_tuning


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args()


def _save_class_mapping(path: Path, class_names: tuple[str, ...]) -> None:
    path.write_text(
        json.dumps(
            {str(index): name for index, name in enumerate(class_names)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    tf.keras.utils.set_random_seed(DATA_CONFIG.seed)
    splits = load_dataset_splits(args.data_dir)
    model = create_model(len(splits.class_names))
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.artifacts_dir / MODEL_CONFIG.metrics_filename
    callbacks = [
        tf.keras.callbacks.CSVLogger(metrics_path, append=False),
    ]

    model.fit(
        splits.train,
        validation_data=splits.validation,
        epochs=TRAINING_CONFIG.epochs,
        verbose=TRAINING_CONFIG.verbose,
        callbacks=callbacks,
    )
    if TRAINING_CONFIG.fine_tune_epochs > 0:
        enable_fine_tuning(model)
        fine_tuning_callbacks = [
            tf.keras.callbacks.CSVLogger(metrics_path, append=True),
        ]
        model.fit(
            splits.train,
            validation_data=splits.validation,
            initial_epoch=TRAINING_CONFIG.epochs,
            epochs=(
                TRAINING_CONFIG.epochs + TRAINING_CONFIG.fine_tune_epochs
            ),
            verbose=TRAINING_CONFIG.verbose,
            callbacks=fine_tuning_callbacks,
        )

    model.save(args.artifacts_dir / MODEL_CONFIG.artifact_filename)
    _save_class_mapping(
        args.artifacts_dir / MODEL_CONFIG.class_mapping_filename,
        splits.class_names,
    )
    print(f"Saved model to {args.artifacts_dir / MODEL_CONFIG.artifact_filename}")
    print(
        "Saved class mapping to "
        f"{args.artifacts_dir / MODEL_CONFIG.class_mapping_filename}"
    )


if __name__ == "__main__":
    main()
