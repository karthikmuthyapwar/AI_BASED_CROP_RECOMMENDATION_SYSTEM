"""SoilNET loader utilities used by Backend.py."""

from __future__ import annotations

from pathlib import Path

from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential, model_from_json


ROOT_DIR = Path(__file__).resolve().parent
MODEL_JSON_CANDIDATES = [
    ROOT_DIR / "Saved Model" / "SoilNET model.json",
    ROOT_DIR / "model.json",
]
MODEL_WEIGHTS_CANDIDATES = [
    ROOT_DIR / "Saved Model" / "weights.h5",
    ROOT_DIR / "Saved Model" / "soilnet_weights.h5",
    ROOT_DIR / "weights.h5",
]


CUSTOM_OBJECTS = {
    "Sequential": Sequential,
    "InputLayer": InputLayer,
    "Conv2D": Conv2D,
    "Activation": Activation,
    "MaxPooling2D": MaxPooling2D,
    "Dropout": Dropout,
    "Flatten": Flatten,
    "Dense": Dense,
    "BatchNormalization": BatchNormalization,
}


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_model():
    """Load serialized legacy SoilNET architecture and weights."""
    model_json_path = _first_existing(MODEL_JSON_CANDIDATES)
    if model_json_path is None:
        raise FileNotFoundError(
            "Could not find SoilNET model JSON in: "
            + ", ".join(str(path) for path in MODEL_JSON_CANDIDATES)
        )

    with model_json_path.open("r", encoding="utf-8") as json_file:
        loaded_model_json = json_file.read()

    # Custom objects keep compatibility with legacy Keras 2.4 JSON exports.
    loaded_model = model_from_json(loaded_model_json, custom_objects=CUSTOM_OBJECTS)

    for weights_path in MODEL_WEIGHTS_CANDIDATES:
        if not weights_path.exists():
            continue
        try:
            loaded_model.load_weights(str(weights_path))
            break
        except Exception:
            # Keep architecture-only model if weight format is incompatible.
            continue

    return loaded_model


model = load_model()
