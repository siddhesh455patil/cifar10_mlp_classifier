import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main():

    results_path = (
        "outputs/tuning_results.csv"
    )

    if not os.path.exists(
        results_path
    ):
        raise FileNotFoundError(
            "Run training/tune.py first."
        )

    results = pd.read_csv(
        results_path
    )

    results = results.sort_values(
        "validation_macro_f1",
        ascending=False
    )

    print("\nHyperparameter results:")
    print(results.to_string(index=False))

    plt.figure(
        figsize=(12, 7)
    )

    labels = [
        (
            f"{row.hidden_layers}\n"
            f"LR={row.learning_rate}\n"
            f"Batch={int(row.batch_size)}"
        )
        for _, row in results.iterrows()
    ]

    scores = results[
        "validation_macro_f1"
    ]

    plt.bar(
        range(len(scores)),
        scores
    )

    plt.xticks(
        range(len(scores)),
        labels,
        rotation=90
    )

    plt.ylabel(
        "Validation Macro F1"
    )

    plt.xlabel(
        "Hyperparameter Configuration"
    )

    plt.title(
        "Hyperparameter Tuning Results"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/tuning_results.png",
        dpi=150
    )

    plt.close()


if __name__ == "__main__":
    main()