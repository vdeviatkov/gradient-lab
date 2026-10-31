import random

import numpy as np
import pytest
from ml_scratch.numpy.mlp_mnist import (
    Activation,
    ClassificationNetwork,
    DenseLayer,
    Loss,
    NetworkTrainingConfig,
    byte_images,
)
from ml_scratch.numpy.mlp_mnist import model as numpy_model
from ml_scratch.numpy.mlp_mnist.experiment import run_experiment
from ml_scratch.pure.mlp_mnist import ClassificationNetwork as ScalarNetwork
from ml_scratch.pure.mlp_mnist.experiment import run_experiment as scalar_experiment
from ml_scratch.pure.softmax_mnist.experiment import sparse_mnist

from ml_scratch.pure.mnist import FILES, mnist_directory

HAS_MNIST = all((mnist_directory() / name).is_file() for name in FILES)
ID, S, T, R = Activation.IDENTITY, Activation.SIGMOID, Activation.TANH, Activation.RELU
SOFTMAX = Loss.SOFTMAX_CROSS_ENTROPY
ARCHITECTURES = {
    "softmax only": (DenseLayer(23, 4),),
    "one ReLU layer": (DenseLayer(23, 9, R), DenseLayer(9, 4)),
    "tanh then sigmoid": (DenseLayer(23, 6, T), DenseLayer(6, 5, S), DenseLayer(5, 4)),
}


def byte_problem(count=150, width=23, classes=4, seed=2):
    """Random byte images where class k brightens feature k; odd sizes exercise every edge."""
    rng = random.Random(seed)
    labels = [rng.randrange(classes) for _ in range(count)]
    images = []
    for label in labels:
        image = [rng.choice((0, 0, rng.randrange(256))) for _ in range(width)]
        image[label] = 255
        images.append(bytes(image))
    return images, labels


@pytest.mark.parametrize("layers", ARCHITECTURES.values(), ids=ARCHITECTURES.keys())
def test_training_and_evaluation_match_the_scalar_network(layers):
    images, labels = byte_problem()
    data, sparse = byte_images(images, labels, 23), sparse_mnist(images, labels, 23)
    network, scalar = ClassificationNetwork(layers, SOFTMAX, 6), ScalarNetwork(layers, SOFTMAX, 6)
    np.testing.assert_array_equal(network.parameters(), scalar.parameters())
    np.testing.assert_allclose(
        network.gradient(data), scalar.gradient(sparse), rtol=1e-12, atol=1e-16
    )
    config = NetworkTrainingConfig(learning_rate=0.5, max_epochs=4, batch_size=16, seed=7)
    result, reference = network.fit(data, config), scalar.fit(sparse, config)
    np.testing.assert_allclose(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-12)
    np.testing.assert_allclose(network.parameters(), scalar.parameters(), rtol=1e-11, atol=1e-14)
    assert network.loss(data) == pytest.approx(scalar.loss(sparse), rel=1e-12)
    np.testing.assert_array_equal(network.predictions(data), scalar.predictions(sparse))
    np.testing.assert_array_equal(network.confusion_matrix(data), scalar.confusion_matrix(sparse))
    assert network.accuracy(data) == scalar.accuracy(sparse)


def test_chunked_evaluation_matches_one_pass(monkeypatch):
    images, labels = byte_problem(count=50)
    data = byte_images(images, labels, 23)
    network = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 1)
    network.fit(data, NetworkTrainingConfig(max_epochs=2, batch_size=8))
    whole = (network.loss(data), network.predictions(data), network.gradient(data))
    monkeypatch.setattr(numpy_model, "CHUNK", 7)
    assert network.loss(data) == pytest.approx(whole[0], rel=1e-14)
    np.testing.assert_array_equal(network.predictions(data), whole[1])
    np.testing.assert_allclose(network.gradient(data), whole[2], rtol=1e-12, atol=1e-17)


def test_plain_examples_checkpoints_and_reproducibility(tmp_path):
    images, labels = byte_problem(count=40)
    data = byte_images(images, labels, 23)
    examples = [
        (np.frombuffer(image, np.uint8) / 255.0, label) for image, label in zip(images, labels)
    ]
    network = ClassificationNetwork(ARCHITECTURES["tanh then sigmoid"], SOFTMAX, 3)
    assert network.loss(examples) == network.loss(data)
    config = NetworkTrainingConfig(learning_rate=0.3, max_epochs=3, batch_size=6, seed=4)
    twin = ClassificationNetwork(ARCHITECTURES["tanh then sigmoid"], SOFTMAX, 3)
    assert network.fit(data, config) == twin.fit(examples, config)
    np.testing.assert_array_equal(network.parameters(), twin.parameters())
    path = tmp_path / "mlp.checkpoint"
    network.save(path)
    restored = ClassificationNetwork.load(path)
    assert type(restored) is ClassificationNetwork
    np.testing.assert_array_equal(restored.parameters(), network.parameters())
    # The file format is shared, so the scalar network restores the same function.
    assert ScalarNetwork.load(path).parameters() == network.parameters().tolist()


def test_divergence_and_invalid_input():
    images, labels = byte_problem(count=20)
    data = byte_images(images, labels, 23)
    network = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 2)
    with pytest.raises(ArithmeticError):
        network.fit(data, NetworkTrainingConfig(learning_rate=1e300, max_epochs=3))
    with pytest.raises(ValueError):
        ClassificationNetwork(ARCHITECTURES["one ReLU layer"], Loss.BINARY_CROSS_ENTROPY)
    for dataset in (
        [],
        [([0.0] * 23, 4)],
        [([0.0] * 23, -1)],
        [([0.0] * 22, 1)],
        [([np.nan] + [0.0] * 22, 1)],
        byte_images([bytes(5)], [0], 5),
    ):
        with pytest.raises(ValueError):
            network.loss(dataset)


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST is not downloaded; see scripts/download_mnist.sh")
def test_real_mnist_subset_matches_the_scalar_protocol(tmp_path):
    options = {
        "max_samples": 1200,
        "search_epochs": 1,
        "final_epochs": 2,
        "linear_epochs": 2,
        "validation_size": 300,
        "check": False,
    }
    result = run_experiment(checkpoint=tmp_path / "numpy.checkpoint", **options)
    reference = scalar_experiment(checkpoint=tmp_path / "scalar.checkpoint", **options)
    for ours, theirs in zip(result.candidates, reference.candidates, strict=True):
        assert ours.validation_accuracy == theirs.validation_accuracy
        assert ours.final_loss == pytest.approx(theirs.final_loss, rel=1e-11)
    assert result.selected.name == reference.selected.name
    np.testing.assert_allclose(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-11)
    assert result.validation_per_epoch == reference.validation_per_epoch
    assert result.mlp_confusion == reference.mlp_confusion
    assert result.linear_confusion == reference.linear_confusion
