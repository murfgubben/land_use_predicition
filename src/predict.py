"""Command-line image prediction entry point."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import ARTIFACTS_DIR, MODEL_CONFIG, OUTPUTS_DIR
    from src.data import preprocess_image, preprocess_image_bytes
    from src.visualize import plot_confidence_scores, plot_input_comparison
else:
    from .config import ARTIFACTS_DIR, MODEL_CONFIG, OUTPUTS_DIR
    from .data import preprocess_image, preprocess_image_bytes
    from .visualize import plot_confidence_scores, plot_input_comparison


def _load_artifacts(
    artifacts_dir: Path,
) -> tuple[tf.keras.Model, tuple[str, ...]]:
    model_path = artifacts_dir / MODEL_CONFIG.artifact_filename
    mapping_path = artifacts_dir / MODEL_CONFIG.class_mapping_filename
    missing = [
        str(path) for path in (model_path, mapping_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required prediction artifact(s) not found: " + ", ".join(missing)
        )

    try:
        model = tf.keras.models.load_model(model_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Could not load trained model from {model_path}: {error}"
        ) from error

    try:
        class_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        class_names = tuple(
            class_mapping[str(index)] for index in range(len(class_mapping))
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            f"Could not read valid class mapping from {mapping_path}: {error}"
        ) from error
    return model, class_names


def predict_processed_image(
    image: tf.Tensor,
    model: tf.keras.Model,
    class_names: tuple[str, ...],
    source_name: str = "image",
) -> dict[str, float]:
    """Predict from a preprocessed image using an already loaded model."""
    try:
        output = np.asarray(model(tf.expand_dims(image, axis=0), training=False))
    except (tf.errors.InvalidArgumentError, ValueError) as error:
        raise RuntimeError(
            f"Could not run inference for {source_name}: {error}"
        ) from error

    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError(
            "Model prediction must have shape [1, number_of_classes], "
            f"got {output.shape}."
        )
    if output.shape[1] != len(class_names):
        raise ValueError(
            "Model output count does not match class mapping: "
            f"{output.shape[1]} != {len(class_names)}."
        )
    probabilities = output[0].astype(float)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite prediction probabilities.")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
        probabilities = tf.nn.softmax(output[0]).numpy().astype(float)

    ranked = sorted(
        zip(class_names, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    return {name: float(probability) for name, probability in ranked}


def predict(image_path: str) -> dict[str, float]:
    """Return sorted class probabilities for a JPEG, PNG, or supported TIFF."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        image = preprocess_image(path)
    except (OSError, ValueError, tf.errors.InvalidArgumentError) as error:
        raise ValueError(f"Could not decode image {path}: {error}") from error
    model, class_names = _load_artifacts(ARTIFACTS_DIR)
    return predict_processed_image(image, model, class_names, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        probabilities = predict(str(args.image_path))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Prediction failed: {error}") from error
    for class_name, probability in probabilities.items():
        print(f"{class_name:<24} {probability:.6f} ({probability:.2%})")
    try:
        processed_image = preprocess_image(args.image_path).numpy()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        comparison_path = OUTPUTS_DIR / "comparison.png"
        confidence_path = OUTPUTS_DIR / "confidence.png"
        plot_input_comparison(args.image_path, processed_image, comparison_path)
        plot_confidence_scores(probabilities, confidence_path)
    except (OSError, ValueError, tf.errors.InvalidArgumentError) as error:
        raise SystemExit(f"Could not save prediction visualizations: {error}") from error
    print(f"\nSaved input comparison to {comparison_path}")
    print(f"Saved confidence chart to {confidence_path}")


if __name__ == "__main__":
    main()
