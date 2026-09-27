import json
import os

import numpy as np

from model.activations import relu, softmax
from model.backpropagation import backward
from model.initialization import he_initialization
from model.losses import categorical_cross_entropy
from model.optimizer import gradient_descent_update


class MLP:
    """
    Multilayer Perceptron implemented using NumPy.

    Example architecture:
        3072 -> 256 -> 128 -> 10
    """

    def __init__(
        self,
        input_size=3072,
        hidden_layers=(256, 128),
        output_size=10,
        seed=42
    ):
        self.input_size = int(input_size)
        self.hidden_layers = tuple(
            int(size) for size in hidden_layers
        )
        self.output_size = int(output_size)

        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("Layer sizes must be positive.")

        if not self.hidden_layers:
            raise ValueError("At least one hidden layer is required.")

        if any(size <= 0 for size in self.hidden_layers):
            raise ValueError("Hidden layer sizes must be positive.")

        self.rng = np.random.default_rng(seed)

        layer_sizes = [
            self.input_size,
            *self.hidden_layers,
            self.output_size
        ]

        self.parameters = {}

        for layer in range(1, len(layer_sizes)):
            n_in = layer_sizes[layer - 1]
            n_out = layer_sizes[layer]

            self.parameters[f"W{layer}"] = he_initialization(
                n_in,
                n_out,
                self.rng
            )

            self.parameters[f"b{layer}"] = np.zeros(
                (1, n_out),
                dtype=np.float32
            )

        self.num_layers = len(layer_sizes) - 1

    def forward(self, x):
        """
        Perform forward propagation.

        Returns
        -------
        probabilities : ndarray
        cache : dict
        """

        activation = np.asarray(x, dtype=np.float32)
        cache = {}

        for layer in range(1, self.num_layers):
            z = (
                activation @ self.parameters[f"W{layer}"]
                + self.parameters[f"b{layer}"]
            )

            activation = relu(z)

            cache[f"z{layer}"] = z
            cache[f"a{layer}"] = activation

        output_layer = self.num_layers

        z_output = (
            activation @ self.parameters[f"W{output_layer}"]
            + self.parameters[f"b{output_layer}"]
        )

        probabilities = softmax(z_output)

        cache[f"z{output_layer}"] = z_output
        cache[f"a{output_layer}"] = probabilities

        return probabilities, cache

    def loss(self, x, y_one_hot):
        """
        Calculate cross-entropy loss.
        """

        probabilities, _ = self.forward(x)

        return categorical_cross_entropy(
            y_one_hot,
            probabilities
        )

    def train_batch(
        self,
        x,
        y_one_hot,
        learning_rate=0.001
    ):
        """
        Perform one mini-batch training step.
        """

        probabilities, cache = self.forward(x)

        batch_loss = categorical_cross_entropy(
            y_one_hot,
            probabilities
        )

        gradients = backward(
            x,
            self.parameters,
            cache,
            y_one_hot
        )

        for layer in range(1, self.num_layers + 1):
            self.parameters[f"W{layer}"], \
            self.parameters[f"b{layer}"] = gradient_descent_update(
                self.parameters[f"W{layer}"],
                self.parameters[f"b{layer}"],
                (
                    gradients[f"dW{layer}"],
                    gradients[f"db{layer}"]
                ),
                learning_rate
            )

        return float(batch_loss)

    def predict_proba(self, x, batch_size=512):
        """
        Return class probabilities in batches.

        Batching avoids creating unnecessarily large
        intermediate arrays during inference.
        """

        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input shape (N, {self.input_size})."
            )

        if len(x) == 0:
            return np.empty(
                (0, self.output_size),
                dtype=np.float32
            )

        predictions = []

        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size]

            probabilities, _ = self.forward(batch)

            predictions.append(probabilities)

        return np.concatenate(predictions, axis=0)

    def predict(self, x, batch_size=512):
        """
        Return predicted class indices.
        """

        probabilities = self.predict_proba(
            x,
            batch_size=batch_size
        )

        return np.argmax(probabilities, axis=1)

    def config(self):
        """
        Return the model configuration.
        """

        return {
            "input_size": self.input_size,
            "hidden_layers": list(self.hidden_layers),
            "output_size": self.output_size
        }

    def save(self, directory):
        """
        Save model configuration and weights.
        """

        os.makedirs(directory, exist_ok=True)

        weights_path = os.path.join(
            directory,
            "model_weights.npz"
        )

        config_path = os.path.join(
            directory,
            "model_config.json"
        )

        np.savez(
            weights_path,
            **self.parameters
        )

        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(self.config(), file, indent=4)

        print(f"Model saved to: {directory}")

    @classmethod
    def load(cls, directory):
        """
        Load a previously saved model.
        """

        config_path = os.path.join(
            directory,
            "model_config.json"
        )

        weights_path = os.path.join(
            directory,
            "model_weights.npz"
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(config_path)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(weights_path)

        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        model = cls(
            input_size=config["input_size"],
            hidden_layers=config["hidden_layers"],
            output_size=config["output_size"]
        )

        with np.load(weights_path) as saved_weights:
            for key in model.parameters:
                model.parameters[key] = saved_weights[key].copy()

        return model