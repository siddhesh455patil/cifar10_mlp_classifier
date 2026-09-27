import json
import os

import numpy as np

from model.mlp import MLP
from preprocessing.dataset_loader import (
    load_cifar10,
    CLASS_NAMES
)
from preprocessing.image_preprocessing import (
    preprocess_array
)
from evaluation.metrics import calculate_metrics
from evaluation.visualization import (
    plot_confusion_matrix,
    plot_misclassified_images
)


def main():

    os.makedirs(
        "outputs/evaluation",
        exist_ok=True
    )

    print("Loading CIFAR-10 test set...")

    _, _, X_test, y_test = load_cifar10()

    print(
        f"Test set size: {len(X_test)}"
    )

    print("Preprocessing test images...")

    X_test_processed = np.stack([
        preprocess_array(image)
        for image in X_test
    ])

    print("Loading final trained model...")

    model = MLP.load(
        "saved_model/final"
    )

    print("Generating predictions...")

    probabilities = model.predict_proba(
        X_test_processed,
        batch_size=512
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    metrics, report, matrix = calculate_metrics(
        y_true=y_test,
        y_pred=predictions,
        class_names=CLASS_NAMES
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    with open(
        "outputs/evaluation/metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    with open(
        "outputs/evaluation/classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    np.savetxt(
        "outputs/evaluation/confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d"
    )

    plot_confusion_matrix(
        matrix,
        CLASS_NAMES,
        "outputs/evaluation/confusion_matrix.png"
    )

    plot_misclassified_images(
        X_test,
        y_test,
        predictions,
        probabilities,
        CLASS_NAMES,
        "outputs/evaluation/misclassified_images.png"
    )

    print(
        "\nEvaluation results saved in "
        "outputs/evaluation/"
    )


if __name__ == "__main__":
    main()