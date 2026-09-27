import os
import pickle
import tarfile
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def download_cifar10(data_dir="data/raw"):
    """
    Download and extract the CIFAR-10 Python dataset.

    Returns
    -------
    str
        Path to the extracted CIFAR-10 directory.
    """

    os.makedirs(data_dir, exist_ok=True)

    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
    archive_path = os.path.join(data_dir, "cifar-10-python.tar.gz")

    if os.path.exists(extracted_dir):
        print("CIFAR-10 dataset already exists.")
        return extracted_dir

    print("Downloading CIFAR-10 dataset...")

    urllib.request.urlretrieve(
        CIFAR10_URL,
        archive_path
    )

    print("Extracting CIFAR-10 dataset...")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)

    print("CIFAR-10 dataset ready.")

    return extracted_dir


def _load_batch(file_path):
    """
    Load one CIFAR-10 batch file.
    """

    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    return images, labels


def load_cifar10(data_dir="data/raw"):
    """
    Load the complete CIFAR-10 dataset.

    Returns
    -------
    X_train : np.ndarray
        Shape: (50000, 32, 32, 3)

    y_train : np.ndarray
        Shape: (50000,)

    X_test : np.ndarray
        Shape: (10000, 32, 32, 3)

    y_test : np.ndarray
        Shape: (10000,)
    """

    dataset_dir = download_cifar10(data_dir)

    train_images = []
    train_labels = []

    for i in range(1, 6):

        batch_path = os.path.join(
            dataset_dir,
            f"data_batch_{i}"
        )

        images, labels = _load_batch(batch_path)

        train_images.append(images)
        train_labels.extend(labels)

    test_path = os.path.join(
        dataset_dir,
        "test_batch"
    )

    test_images, test_labels = _load_batch(test_path)

    X_train = np.concatenate(train_images, axis=0)
    y_train = np.array(train_labels)

    X_test = test_images
    y_test = np.array(test_labels)

    # CIFAR-10 stores images as:
    # 3072 = 1024 red + 1024 green + 1024 blue
    #
    # Convert to:
    # (N, 32, 32, 3)

    X_train = X_train.reshape(
        -1,
        3,
        32,
        32
    )

    X_test = X_test.reshape(
        -1,
        3,
        32,
        32
    )

    X_train = X_train.transpose(
        0,
        2,
        3,
        1
    )

    X_test = X_test.transpose(
        0,
        2,
        3,
        1
    )

    print(f"Training images: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")

    return X_train, y_train, X_test, y_test


def create_train_validation_split(
    X,
    y,
    validation_size=5000,
    random_state=42
):
    """
    Split the official 50,000 CIFAR-10 training images into:

    45,000 training images
    5,000 validation images

    Stratification preserves class distribution.
    """

    if len(X) != 50000:
        raise ValueError(
            "Expected the official CIFAR-10 training set "
            "containing exactly 50,000 images."
        )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training split: {X_train.shape}")
    print(f"Validation split: {X_validation.shape}")

    return (
        X_train,
        X_validation,
        y_train,
        y_validation
    )