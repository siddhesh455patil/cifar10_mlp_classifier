import json
import os

import numpy as np

from model.losses import categorical_cross_entropy


def one_hot(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.
    """

    labels = np.asarray(labels, dtype=np.int64)

    encoded = np.zeros(
        (len(labels), num_classes),
        dtype=np.float32
    )

    encoded[np.arange(len(labels)), labels] = 1.0

    return encoded


def _evaluate(model, x, y, batch_size=512):
    """
    Evaluate loss and accuracy without updating weights.
    """

    total_loss = 0.0
    total_correct = 0

    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]

        probabilities = model.predict_proba(
            xb,
            batch_size=batch_size
        )

        total_loss += categorical_cross_entropy(
            one_hot(yb, model.output_size),
            probabilities
        ) * len(xb)

        predictions = np.argmax(probabilities, axis=1)

        total_correct += np.sum(predictions == yb)

    return {
        "loss": float(total_loss / len(x)),
        "accuracy": float(total_correct / len(x))
    }


def train_model(
    model,
    x_train,
    y_train,
    x_validation,
    y_validation,
    epochs=30,
    batch_size=128,
    learning_rate=0.001,
    seed=42,
    output_dir=None
):
    """
    Train an MLP using mini-batch gradient descent.

    Returns
    -------
    model : MLP
        Trained model.

    history : dict
        Training and validation metrics per epoch.
    """

    if len(x_train) == 0 or len(x_validation) == 0:
        raise ValueError("Training and validation data cannot be empty.")

    if len(x_train) != len(y_train):
        raise ValueError("Training images and labels do not match.")

    if len(x_validation) != len(y_validation):
        raise ValueError("Validation images and labels do not match.")

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError(
            "Epochs, batch size and learning rate must be positive."
        )

    rng = np.random.default_rng(seed)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    number_of_samples = len(x_train)

    for epoch in range(epochs):
        permutation = rng.permutation(number_of_samples)

        x_shuffled = x_train[permutation]
        y_shuffled = y_train[permutation]

        for start in range(0, number_of_samples, batch_size):
            end = start + batch_size

            xb = x_shuffled[start:end]
            yb = y_shuffled[start:end]

            model.train_batch(
                xb,
                one_hot(yb, model.output_size),
                learning_rate=learning_rate
            )

        train_metrics = _evaluate(
            model,
            x_train,
            y_train
        )

        validation_metrics = _evaluate(
            model,
            x_validation,
            y_validation
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["train_accuracy"].append(
            train_metrics["accuracy"]
        )

        history["validation_loss"].append(
            validation_metrics["loss"]
        )

        history["validation_accuracy"].append(
            validation_metrics["accuracy"]
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train accuracy: {train_metrics['accuracy']:.4f} | "
            f"Val loss: {validation_metrics['loss']:.4f} | "
            f"Val accuracy: {validation_metrics['accuracy']:.4f}"
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            history_path = os.path.join(
                output_dir,
                "history.json"
            )

            with open(
                history_path,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(history, file, indent=4)

    return model, history