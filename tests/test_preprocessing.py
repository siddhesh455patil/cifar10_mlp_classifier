import numpy as np

from preprocessing.image_preprocessing import (
    preprocess_array
)


def test_preprocess_array():

    image = np.zeros(
        (32, 32, 3),
        dtype=np.uint8
    )

    processed = preprocess_array(
        image
    )

    assert processed.shape == (
        3072,
    )

    assert processed.dtype == np.float32

    assert np.all(
        processed >= 0
    )

    assert np.all(
        processed <= 1
    )


def test_preprocess_values():

    image = np.full(
        (32, 32, 3),
        255,
        dtype=np.uint8
    )

    processed = preprocess_array(
        image
    )

    assert np.allclose(
        processed,
        1.0
    )