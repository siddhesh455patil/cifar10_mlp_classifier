import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from model.mlp import MLP
from preprocessing.dataset_loader import (
    load_cifar10,
    create_train_validation_split
)
from preprocessing.image_preprocessing import preprocess_array
from training.train import train_model


def main():
    os.makedirs("outputs", exist_ok=True)

    # Load official training partition only.
    X, y, _, _ = load_cifar10()

    X_train, X_val, y_train, y_val = (
        create_train_validation_split(X, y)
    )

    print("Preprocessing training images...")

    X_train = np.stack([
        preprocess_array(image) for image in X_train
    ])

    X_val = np.stack([
        preprocess_array(image) for image in X_val
    ])

    # Required baseline architecture.
    model = MLP(
        input_size=3072,
        hidden_layers=(256, 128),
        output_size=10,
        seed=42
    )

    model, history = train_model(
        model=model,
        x_train=X_train,
        y_train=y_train,
        x_validation=X_val,
        y_validation=y_val,
        epochs=30,
        batch_size=128,
        learning_rate=0.001,
        seed=42,
        output_dir="outputs/baseline"
    )

    model.save("saved_model/baseline")

    with open(
        "outputs/baseline/history.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(history, file, indent=4)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["validation_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Baseline Model Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/baseline_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training accuracy"
    )
    plt.plot(
        epochs,
        history["validation_accuracy"],
        label="Validation accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline Model Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/baseline_accuracy.png", dpi=150)
    plt.close()

    print("Baseline training completed.")
    print("Model: saved_model/baseline")
    print("Plots: outputs/baseline_loss.png and outputs/baseline_accuracy.png")


if __name__ == "__main__":
    main()