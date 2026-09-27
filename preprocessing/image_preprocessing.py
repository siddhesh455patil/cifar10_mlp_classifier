import io

import numpy as np
from PIL import Image


IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
INPUT_SIZE = 3072


def preprocess_array(image_array):
    """
    Convert a 32x32 RGB image into the MLP input format.

    Steps:
    1. Validate shape.
    2. Convert to float32.
    3. Normalize pixel values from [0,255] to [0,1].
    4. Flatten into 3072 features.
    """

    image_array = np.asarray(image_array)

    if image_array.shape != (32, 32, 3):
        raise ValueError(
            f"Expected image shape (32, 32, 3), "
            f"got {image_array.shape}"
        )

    image_array = image_array.astype(np.float32)

    image_array /= 255.0

    flattened = image_array.reshape(-1)

    if flattened.shape[0] != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE} features, "
            f"got {flattened.shape[0]}"
        )

    return flattened


def preprocess_pil_image(image):
    """
    Preprocess a PIL image for model prediction.
    """

    image = image.convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(image)

    return preprocess_array(image_array)


def preprocess_image_file(file_path):
    """
    Load and preprocess an image from a file path.
    """

    image = Image.open(file_path)

    return preprocess_pil_image(image)


def preprocess_uploaded_bytes(image_bytes):
    """
    Load and preprocess image bytes received
    from a Flask upload.
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    return preprocess_pil_image(image)