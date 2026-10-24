# Uses Keras to load a trained hybrid EfficientNetB0-ViT model
"""Model loading and inference for the EfficientNetB0-ViT classifier."""

import os

import numpy as np
from PIL import Image

_model = None
_model_load_error = None

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "rose_efficientnet_vit_hybrid_best.keras",
)

IMAGE_SIZE = (224, 224)


def load_model():
    """Load and cache the trained hybrid model."""
    global _model, _model_load_error

    if _model is not None:
        return _model

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    try:
        import keras
        import keras_hub  # noqa: F401

        from keras.applications.efficientnet import (
            preprocess_input as efficientnet_preprocess_input,
        )

        # Import the class so Keras can resolve the saved ViT converter.
        from keras_hub.src.models.vit.vit_image_converter import (
            ViTImageConverter,  # noqa: F401
        )

        custom_objects = {
            # The saved Lambda layer refers to this function by name.
            "preprocess_input": efficientnet_preprocess_input,
        }

        _model = keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )

        _model_load_error = None
        return _model

    except Exception as exc:
        _model_load_error = (
            f"Failed to load model from {MODEL_PATH}: {exc}"
        )
        raise RuntimeError(_model_load_error) from exc


def get_model():
    """Compatibility alias for code that calls get_model()."""
    return load_model()


def preprocess_image(image_path):
    """Resize an image and prepare a float32 RGB batch."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        # Keep pixel values in the 0–255 range.
        # Preprocessing is handled by the saved model layers.
        array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(array, axis=0)


def predict_image(image_path):
    """Predict one image and return its class, confidence and scores."""
    from utils.disease_info import CLASS_NAMES

    model = load_model()
    batch = preprocess_image(image_path)

    predictions = np.asarray(model.predict(batch, verbose=0))
    probabilities = predictions.reshape(-1)

    if probabilities.size != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {probabilities.size} scores, "
            f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes."
        )

    # Apply softmax only if its output does not look normalized.
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)
    ):
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned non-finite predictions.")

        shifted = probabilities - np.max(probabilities)
        exp_values = np.exp(shifted)
        probabilities = exp_values / exp_values.sum()

    probabilities = probabilities.astype(np.float32)
    predicted_index = int(np.argmax(probabilities))

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "scores": scores,
    }
