"""
Model loading and inference helper.

Place your trained model at:
    models/rose_hybrid_model_final2_best.keras

The model is loaded once and cached for the lifetime of the process.

Notes
-----
The saved `.keras` file was created with a Lambda layer that wraps
`tensorflow.keras.applications.vgg16.preprocess_input`. Keras 3 does
not know how to resolve that bare name on load, so we provide it via
`custom_objects`. Any other custom callables referenced by the model
can be added to `CUSTOM_OBJECTS` below.
"""

from tensorflow.keras.applications.vgg16 import (  # type: ignore[import]
    preprocess_input as vgg16_preprocess_input,
)

import os

import numpy as np
from PIL import Image

_model = None
_model_load_error = None

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "rose_hybrid_model_final2_best.keras"
)

IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------
# Custom objects required to deserialize the saved model.
# Add here if Keras later complains about another symbol.
# ---------------------------------------------------------------------
def _build_custom_objects():
    """Return dict of custom_objects needed by keras.load_model()."""
    from tensorflow.keras.applications.vgg16 import ( # type: ignore
        preprocess_input as vgg16_preprocess_input,
    )

    custom = {
        # The Lambda layer in the saved model references this by name.
        "preprocess_input": vgg16_preprocess_input,
    }

    # Some .keras files saved with `keras.applications` use the module
    # path "<app>.preprocess_input"; registering the plain name above
    # covers those. If your model references `keras_hub` presets or
    # custom ViT blocks, import them here so Keras can resolve them:
    #
    # try:
    #     import keras_hub  # noqa: F401  # registers keras_hub layers
    # except Exception:
    #     pass

    return custom


def _import_ml_stack():
    global _model_load_error
    try:
        import tensorflow as tf  # noqa: F401
        import keras  # noqa: F401
    except Exception as exc:  # pragma: no cover
        _model_load_error = (
            "TensorFlow/Keras could not be imported. "
            f"Original error: {exc}"
        )
        raise RuntimeError(_model_load_error) from exc


def load_model():
    """Load and cache the Keras model."""
    global _model, _model_load_error

    if _model is not None:
        return _model

    _import_ml_stack()

    try:
        import keras
    except Exception as exc:  # pragma: no cover
        _model_load_error = f"Keras import failed: {exc}"
        raise RuntimeError(_model_load_error) from exc

    if not os.path.exists(MODEL_PATH):
        _model_load_error = (
            f"Model file not found at {MODEL_PATH}. "
            "Place your trained .keras model at that path."
        )
        raise RuntimeError(_model_load_error)

    try:
        custom_objects = _build_custom_objects()
        _model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False,  # allow Lambda / custom callables
        )
        _model_load_error = None
        return _model
    except Exception as exc:
        _model_load_error = (
            f"Failed to load model from {MODEL_PATH}: {exc}"
        )
        raise RuntimeError(_model_load_error) from exc


# Convenience alias so both names work.
def get_model():
    return load_model()



def preprocess_image(image_path):
    """Load and prepare an image for the model."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype="float32")
    return np.expand_dims(arr, axis=0)



def predict_image(image_path):
    """
    Run inference on a single image.

    Returns dict with keys:
        - predicted_class (str, internal class key)
        - confidence (float, 0..1)
        - scores (dict: class_key -> float)
    """
    from utils.disease_info import CLASS_NAMES

    model = load_model()
    batch = preprocess_image(image_path)

    raw = model.predict(batch)
    raw = np.asarray(raw).reshape(-1)

    if raw.size == len(CLASS_NAMES) and not np.isclose(
        raw.sum(), 1.0, atol=1e-3
    ):
        exp = np.exp(raw - np.max(raw))
        probs = exp / exp.sum()
    else:
        probs = raw

    probs = probs.astype("float32")

    index = int(np.argmax(probs))
    predicted_class = (
        CLASS_NAMES[index]
        if index < len(CLASS_NAMES)
        else f"Class {index}"
    )
    confidence = float(probs[index])

    scores = {
        CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}": float(p)
        for i, p in enumerate(probs)
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "scores": scores,
    }