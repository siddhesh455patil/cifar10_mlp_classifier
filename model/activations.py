import numpy as np


def relu(x):
    """
    ReLU activation function.

    ReLU(x) = max(0, x)
    """

    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU.
    """

    return (x > 0).astype(np.float32)


def softmax(x):
    """
    Numerically stable Softmax activation.

    Used in the output layer for the
    10 CIFAR-10 classes.
    """

    shifted_x = x - np.max(
        x,
        axis=1,
        keepdims=True
    )

    exp_x = np.exp(shifted_x)

    probabilities = (
        exp_x /
        np.sum(
            exp_x,
            axis=1,
            keepdims=True
        )
    )

    return probabilities