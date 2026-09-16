"""Command-line evaluation entry point."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .config import ARTIFACTS_DIR, DATA_DIR, MODEL_CONFIG
from .data import load_dataset_splits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = args.artifacts_dir / MODEL_CONFIG.artifact_filename
    mapping_path = args.artifacts_dir / MODEL_CONFIG.class_mapping_filename
    model = tf.keras.models.load_model(model_path)
    class_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_names = tuple(
        class_mapping[str(index)] for index in range(len(class_mapping))
    )
    splits = load_dataset_splits(args.data_dir)
    if class_names != splits.class_names:
        raise ValueError(
            "Saved class mapping does not match the dataset class mapping: "
            f"{class_names} != {splits.class_names}"
        )

    # One model pass supplies labels and probabilities for every metric.
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch_images, batch_labels in splits.test:
        probabilities.append(model(batch_images, training=False).numpy())
        labels.append(batch_labels.numpy())
    true_labels = np.concatenate(labels)
    predicted_labels = np.argmax(np.concatenate(probabilities), axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=np.arange(len(class_names))
    )
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nPer-class precision/recall:")
    print(report)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)


if __name__ == "__main__":
    main()
