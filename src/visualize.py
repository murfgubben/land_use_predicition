"""Presentable prediction visualizations."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_NAVY = "#183B56"
_TEAL = "#2A9D8F"
_CORAL = "#E76F51"
_PALE_TEAL = "#B8DED8"


def _as_displayable_rgb(image: np.ndarray) -> np.ndarray:
    """Convert common model-image ranges into displayable RGB pixels."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            "Processed image must have shape [height, width, 3], "
            f"got {image.shape}."
        )
    image = image.astype(np.float32)
    minimum = float(np.nanmin(image))
    maximum = float(np.nanmax(image))
    if not np.isfinite([minimum, maximum]).all():
        raise ValueError("Processed image contains non-finite values.")
    if minimum >= -1.01 and maximum <= 1.01 and minimum < 0:
        image = (image + 1.0) * 127.5
    elif minimum >= 0 and maximum <= 1.01:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def plot_input_comparison(
    original_image_path: str | Path,
    processed_image_array: np.ndarray,
    save_path: str | Path,
) -> None:
    """Save a side-by-side view of the original and processed model input."""
    original_path = Path(original_image_path)
    output_path = Path(save_path)
    if not original_path.is_file():
        raise FileNotFoundError(f"Original image not found: {original_path}")
    try:
        original = plt.imread(original_path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not read original image {original_path}: {error}"
        ) from error

    processed = _as_displayable_rgb(np.asarray(processed_image_array))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    try:
        axes[0].imshow(original)
        axes[0].set_title("Your image", color=_NAVY, fontsize=14, weight="bold")
        axes[1].imshow(processed)
        axes[1].set_title(
            "What the model sees", color=_NAVY, fontsize=14, weight="bold"
        )
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_confidence_scores(
    probabilities: dict[str, float],
    save_path: str | Path,
    top_n: int | None = None,
) -> None:
    """Save a horizontal probability chart, highlighting the top prediction."""
    if not probabilities:
        raise ValueError("At least one class probability is required.")
    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n must be positive or None, got {top_n}.")

    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    names = [item[0] for item in ranked][::-1]
    values = np.asarray([item[1] for item in ranked][::-1], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Probabilities must be finite, non-negative numbers.")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_height = max(4.0, 0.62 * len(names) + 1.2)
    figure, axis = plt.subplots(figsize=(9, figure_height), dpi=150)
    try:
        colors = [_TEAL] * len(names)
        colors[-1] = _CORAL
        bars = axis.barh(names, values, color=colors, height=0.62)
        axis.set_xlim(0, max(1.0, float(values.max()) * 1.18))
        axis.set_xlabel("Confidence", color=_NAVY, fontsize=11)
        axis.set_title(
            "Prediction confidence", color=_NAVY, fontsize=16, weight="bold",
            pad=16,
        )
        axis.tick_params(axis="y", labelsize=10, colors=_NAVY, length=0)
        axis.tick_params(axis="x", labelsize=9, colors="#5B7083")
        axis.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )
        axis.grid(False)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(_PALE_TEAL)
        for bar, value in zip(bars, values):
            axis.text(
                min(value + 0.01, axis.get_xlim()[1] * 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1%}",
                va="center",
                ha="left",
                color=_NAVY,
                fontsize=10,
                weight="bold" if value == values[-1] else "normal",
            )
        figure.tight_layout()
        figure.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
