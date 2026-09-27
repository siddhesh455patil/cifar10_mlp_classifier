import numpy as np


def categorical_cross_entropy(
    y_true,
    y_pred
):
    """
    Calculate categorical cross-entropy loss.

    y_true:
        One-hot encoded labels.

    y_pred:
        Predicted probabilities.
    """

    epsilon = 1e-12

    y_pred = np.clip(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    loss = -np.sum(
        y_true * np.log(y_pred),
        axis=1
    )

    return np.mean(loss)


def softmax_cross_entropy_gradient(
    y_true,
    y_pred
):
    """
    Gradient of Softmax + Cross Entropy.

    dL/dz = (y_pred - y_true) / batch_size
    """

    batch_size = y_true.shape[0]

    return (
        y_pred - y_true
    ) / batch_size