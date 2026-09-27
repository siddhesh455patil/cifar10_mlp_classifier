def gradient_descent_update(
    weights,
    biases,
    gradients,
    learning_rate
):
    """
    Apply standard gradient descent.
    """

    d_weights, d_biases = gradients

    weights -= learning_rate * d_weights
    biases -= learning_rate * d_biases

    return weights, biases