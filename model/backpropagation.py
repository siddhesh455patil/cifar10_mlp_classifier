import numpy as np

from model.activations import relu_derivative


def backward(x, parameters, cache, y_true):
    """
    Perform backpropagation for an MLP using ReLU hidden
    activations and Softmax + cross-entropy output.

    Parameters
    ----------
    x : ndarray, shape (batch_size, input_size)
        Input batch.

    parameters : dict
        Dictionary containing W1, b1, W2, b2, ..., WL, bL.

    cache : dict
        Activations and pre-activations from forward propagation.

    y_true : ndarray, shape (batch_size, num_classes)
        One-hot encoded labels.

    Returns
    -------
    gradients : dict
        Gradients for all weights and biases.
    """

    batch_size = x.shape[0]

    # Number of affine layers, including output layer.
    num_layers = len(parameters) // 2

    gradients = {}

    # Softmax + cross-entropy gradient.
    output = cache[f"a{num_layers}"]
    dz = (output - y_true) / batch_size

    # Work backwards from output layer to first layer.
    for layer in range(num_layers, 0, -1):
        if layer == 1:
            previous_activation = x
        else:
            previous_activation = cache[f"a{layer - 1}"]

        gradients[f"dW{layer}"] = (
            previous_activation.T @ dz
        )

        gradients[f"db{layer}"] = np.sum(
            dz,
            axis=0,
            keepdims=True
        )

        # Propagate gradients to the preceding hidden layer.
        if layer > 1:
            da_previous = dz @ parameters[f"W{layer}"].T

            z_previous = cache[f"z{layer - 1}"]

            dz = da_previous * relu_derivative(z_previous)

    return gradients