"""FastAPI service for satellite image prediction."""

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.predict import (
        _load_artifacts,
        predict_processed_image,
    )
    from src.data import preprocess_image_bytes
    from src.visualize import confidence_scores_png
    from src.config import ARTIFACTS_DIR
else:
    from .predict import _load_artifacts, predict_processed_image
    from .data import preprocess_image_bytes
    from .visualize import confidence_scores_png
    from .config import ARTIFACTS_DIR


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "image/tiff": {".tif", ".tiff"},
}
ALLOWED_SUFFIXES = {
    suffix
    for suffixes in ALLOWED_CONTENT_TYPES.values()
    for suffix in suffixes
}


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


async def _read_upload(upload: UploadFile) -> tuple[bytes, str]:
    suffix = Path(upload.filename or "").suffix.lower()
    if (
        upload.content_type not in ALLOWED_CONTENT_TYPES
        or suffix not in ALLOWED_SUFFIXES
    ):
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, and TIFF image uploads are supported.",
        )
    if suffix not in ALLOWED_CONTENT_TYPES[upload.content_type]:
        raise HTTPException(
            status_code=415,
            detail="The file extension does not match its declared image type.",
        )

    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image upload exceeds the 10 MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks), suffix


def _predict_upload(
    contents: bytes,
    suffix: str,
    model: tf.keras.Model,
    class_names: tuple[str, ...],
) -> tuple[dict[str, float], tf.Tensor]:
    try:
        processed = preprocess_image_bytes(contents, suffix)
    except (ValueError, OSError, tf.errors.InvalidArgumentError) as error:
        raise HTTPException(
            status_code=400, detail=f"Could not decode uploaded image: {error}"
        ) from error
    try:
        probabilities = predict_processed_image(
            processed, model, class_names, "uploaded image"
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return probabilities, processed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model, class_names = _load_artifacts(ARTIFACTS_DIR)
    app.state.model = model
    app.state.class_names = class_names
    yield


app = FastAPI(
    title="Satellite Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
) -> PredictionResponse:
    contents, suffix = await _read_upload(file)
    probabilities, _ = _predict_upload(
        contents, suffix, app.state.model, app.state.class_names
    )
    predicted_class, confidence = next(iter(probabilities.items()))
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
    )


@app.post("/predict/visualize")
async def predict_visualize_endpoint(
    file: UploadFile = File(...),
) -> StreamingResponse:
    contents, suffix = await _read_upload(file)
    probabilities, _ = _predict_upload(
        contents, suffix, app.state.model, app.state.class_names
    )
    try:
        chart = confidence_scores_png(probabilities)
    except (ValueError, OSError) as error:
        raise HTTPException(
            status_code=500, detail=f"Could not generate visualization: {error}"
        ) from error
    return StreamingResponse(
        BytesIO(chart),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=confidence.png"},
    )
