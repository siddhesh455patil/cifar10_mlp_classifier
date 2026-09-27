import numpy as np


def he_initialization(
    n_in,
    n_out,
    rng
):
    """
    He initialization.

    Suitable for networks using ReLU
    activation functions.
    """

    std = np.sqrt(
        2.0 / n_in
    )

    weights = rng.normal(
        loc=0.0,
        scale=std,
        size=(n_in, n_out)
    ).astype(np.float32)

    return weights