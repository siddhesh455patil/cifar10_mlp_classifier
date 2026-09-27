import argparse
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


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train final CIFAR-10 MLP model."
    )

    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[256, 128],
        help="Hidden layer sizes."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    os.makedirs(
        "outputs/final",
        exist_ok=True
    )

    print("Loading CIFAR-10...")

    X, y, _, _ = load_cifar10()

    X_train, X_val, y_train, y_val = (
        create_train_validation_split(
            X,
            y
        )
    )

    print("Preprocessing training images...")

    X_train = np.stack([
        preprocess_array(image)
        for image in X_train
    ])

    print("Preprocessing validation images...")

    X_val = np.stack([
        preprocess_array(image)
        for image in X_val
    ])

    print("\nFinal model configuration:")

    print(
        f"Architecture: "
        f"3072 -> {' -> '.join(map(str, args.hidden))} -> 10"
    )

    print(
        f"Learning rate: {args.learning_rate}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    model = MLP(
        input_size=3072,
        hidden_layers=tuple(args.hidden),
        output_size=10,
        seed=42
    )

    model, history = train_model(
        model=model,
        x_train=X_train,
        y_train=y_train,
        x_validation=X_val,
        y_validation=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=42,
        output_dir="outputs/final"
    )

    model.save(
        "saved_model/final"
    )

    with open(
        "outputs/final/history.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # Loss graph
    plt.figure()

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")

    plt.title(
        "Final MLP Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/final_loss.png",
        dpi=150
    )

    plt.close()

    # Accuracy graph
    plt.figure()

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["validation_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "Final MLP Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/final_accuracy.png",
        dpi=150
    )

    plt.close()

    print("\nFinal model training completed.")

    print(
        "Model saved in: saved_model/final"
    )


if __name__ == "__main__":
    main()