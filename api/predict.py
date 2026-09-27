import io
import os

import numpy as np
from PIL import Image

from model.mlp import MLP
from preprocessing.image_preprocessing import preprocess_pil_image


MODEL_DIRECTORY = "saved_model/final"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

_model = None


def get_model():
    global _model

    if _model is None:
        _model = MLP.load(MODEL_DIRECTORY)

    return _model


def predict_image(image):
    """
    Predict a PIL Image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.

    Returns
    -------
    list
        Top-3 predictions with class name and confidence.
    """

    model = get_model()

    features = preprocess_pil_image(image)

    # Add batch dimension: (3072,) -> (1, 3072)
    features = np.expand_dims(features, axis=0)

    probabilities = model.predict_proba(features)[0]

    top_indices = np.argsort(probabilities)[::-1][:3]

    predictions = []

    for index in top_indices:
        predictions.append(
            {
                "class": CLASS_NAMES[int(index)],
                "confidence": float(probabilities[index]),
            }
        )

    return predictions


def predict_image_bytes(image_bytes):
    """
    Convert uploaded image bytes into a PIL Image
    before sending it to the preprocessing pipeline.
    """

    image = Image.open(io.BytesIO(image_bytes))

    # Make sure the image is fully loaded before
    # the underlying BytesIO object disappears.
    image.load()

    return predict_image(image)


def predict_uploaded_file(file_storage):
    """
    Predict directly from a Flask uploaded file.
    """

    image = Image.open(file_storage.stream)

    image.load()

    return predict_image(image)


def is_allowed_file(filename):
    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS