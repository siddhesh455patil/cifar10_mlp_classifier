import numpy as np

from model.mlp import MLP


def test_model_output_shape():

    model = MLP(
        input_size=3072,
        hidden_layers=(256, 128),
        output_size=10
    )

    rng = np.random.default_rng(42)

    X = rng.random(
        (8, 3072),
        dtype=np.float32
    )

    probabilities = model.predict_proba(
        X
    )

    assert probabilities.shape == (
        8,
        10
    )


def test_probabilities_sum_to_one():

    model = MLP()

    rng = np.random.default_rng(42)

    X = rng.random(
        (8, 3072),
        dtype=np.float32
    )

    probabilities = model.predict_proba(
        X
    )

    sums = probabilities.sum(
        axis=1
    )

    assert np.allclose(
        sums,
        1.0,
        atol=1e-5
    )


def test_training_changes_weights():

    model = MLP()

    rng = np.random.default_rng(42)

    X = rng.random(
        (8, 3072),
        dtype=np.float32
    )

    labels = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7]
    )

    Y = np.zeros(
        (8, 10),
        dtype=np.float32
    )

    Y[
        np.arange(8),
        labels
    ] = 1.0

    original_weights = (
        model.parameters["W1"].copy()
    )

    model.train_batch(
        X,
        Y,
        learning_rate=0.001
    )

    assert not np.array_equal(
        original_weights,
        model.parameters["W1"]
    )