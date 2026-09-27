import os
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from model.mlp import MLP
from preprocessing.dataset_loader import (
    load_cifar10,
    create_train_validation_split
)
from preprocessing.image_preprocessing import preprocess_array
from training.train import one_hot


def evaluate_validation(model, X, y):
    """
    Calculate validation macro F1 score.
    """

    predictions = model.predict(X)

    score = f1_score(
        y,
        predictions,
        average="macro"
    )

    return float(score)


def run_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_layers,
    learning_rate,
    batch_size,
    epochs=10,
    seed=42
):
    """
    Train one hyperparameter configuration.
    """

    model = MLP(
        input_size=3072,
        hidden_layers=hidden_layers,
        output_size=10,
        seed=seed
    )

    rng = np.random.default_rng(seed)

    number_of_samples = len(X_train)

    for epoch in range(epochs):

        permutation = rng.permutation(
            number_of_samples
        )

        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        for start in range(
            0,
            number_of_samples,
            batch_size
        ):
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            model.train_batch(
                X_batch,
                one_hot(
                    y_batch,
                    num_classes=10
                ),
                learning_rate=learning_rate
            )

        predictions = model.predict(
            X_val
        )

        validation_f1 = f1_score(
            y_val,
            predictions,
            average="macro"
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Validation Macro F1: "
            f"{validation_f1:.4f}"
        )

    final_f1 = evaluate_validation(
        model,
        X_val,
        y_val
    )

    return final_f1


def main():

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    print("Loading CIFAR-10 training data...")

    X, y, _, _ = load_cifar10()

    X_train, X_val, y_train, y_val = (
        create_train_validation_split(
            X,
            y
        )
    )

    print("Preprocessing training data...")

    X_train = np.stack([
        preprocess_array(image)
        for image in X_train
    ])

    print("Preprocessing validation data...")

    X_val = np.stack([
        preprocess_array(image)
        for image in X_val
    ])

    # Three different architecture configurations.
    hidden_layer_configs = [
        (128, 64),
        (256, 128),
        (256, 256)
    ]

    learning_rates = [
        0.001,
        0.0005,
        0.0001
    ]

    batch_sizes = [
        64,
        128,
        256
    ]

    experiments = list(
        itertools.product(
            hidden_layer_configs,
            learning_rates,
            batch_sizes
        )
    )

    print(
        f"Total experiments: {len(experiments)}"
    )

    results = []

    for experiment_number, (
        hidden_layers,
        learning_rate,
        batch_size
    ) in enumerate(
        experiments,
        start=1
    ):

        print("\n" + "=" * 70)

        print(
            f"Experiment "
            f"{experiment_number}/{len(experiments)}"
        )

        print(
            f"Hidden layers: {hidden_layers}"
        )

        print(
            f"Learning rate: {learning_rate}"
        )

        print(
            f"Batch size: {batch_size}"
        )

        print("=" * 70)

        validation_f1 = run_experiment(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=10
        )

        results.append({
            "hidden_layers": str(
                hidden_layers
            ),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "validation_macro_f1": validation_f1
        })

        results_df = pd.DataFrame(results)

        results_df.to_csv(
            "outputs/tuning_results.csv",
            index=False
        )

    print("\nHyperparameter tuning completed.")

    results_df = pd.DataFrame(results)

    best_result = results_df.loc[
        results_df[
            "validation_macro_f1"
        ].idxmax()
    ]

    print("\nBest validation configuration:")
    print(best_result)

    print(
        "\nResults saved to "
        "outputs/tuning_results.csv"
    )


if __name__ == "__main__":
    main()