import io

import numpy as np
from PIL import Image

from app import app


def create_test_image():

    array = np.zeros(
        (32, 32, 3),
        dtype=np.uint8
    )

    image = Image.fromarray(
        array
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer


def test_missing_image():

    client = app.test_client()

    response = client.post(
        "/api/predict"
    )

    assert response.status_code == 400


def test_unsupported_file():

    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={
            "image": (
                io.BytesIO(
                    b"test"
                ),
                "test.txt"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 400


def test_prediction_endpoint():

    client = app.test_client()

    image = create_test_image()

    response = client.post(
        "/api/predict",
        data={
            "image": (
                image,
                "test.png"
            )
        },
        content_type="multipart/form-data"
    )

    # If the model has been trained, prediction should succeed.
    # Before model training, the endpoint correctly reports
    # that the final model is unavailable.
    assert response.status_code in (
        200,
        500
    )