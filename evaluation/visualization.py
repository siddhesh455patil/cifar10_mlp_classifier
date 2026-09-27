import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history,
    output_directory
):
    """
    Plot training/validation loss and accuracy.
    """

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

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
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            "loss_curve.png"
        ),
        dpi=150
    )

    plt.close()

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
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            "accuracy_curve.png"
        ),
        dpi=150
    )

    plt.close()


def plot_confusion_matrix(
    matrix,
    class_names,
    output_path
):
    """
    Plot a confusion matrix.
    """

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "CIFAR-10 Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    threshold = matrix.max() / 2.0

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = matrix[
                row,
                column
            ]

            plt.text(
                column,
                row,
                str(value),
                horizontalalignment="center",
                color="white" if value > threshold else "black"
            )

    plt.ylabel(
        "Actual Class"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def plot_misclassified_images(
    images,
    true_labels,
    predicted_labels,
    probabilities,
    class_names,
    output_path,
    max_images=16
):
    """
    Display incorrectly classified images.
    """

    incorrect_indices = np.where(
        true_labels != predicted_labels
    )[0]

    incorrect_indices = incorrect_indices[
        :max_images
    ]

    if len(incorrect_indices) == 0:
        print(
            "No misclassified images found."
        )
        return

    columns = 4

    rows = int(
        np.ceil(
            len(incorrect_indices) / columns
        )
    )

    plt.figure(
        figsize=(12, 3 * rows)
    )

    for position, index in enumerate(
        incorrect_indices
    ):

        plt.subplot(
            rows,
            columns,
            position + 1
        )

        image = images[index]

        plt.imshow(image)

        true_name = class_names[
            true_labels[index]
        ]

        predicted_name = class_names[
            predicted_labels[index]
        ]

        confidence = probabilities[
            index,
            predicted_labels[index]
        ]

        plt.title(
            f"Actual: {true_name}\n"
            f"Predicted: {predicted_name}\n"
            f"Confidence: {confidence:.2%}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()