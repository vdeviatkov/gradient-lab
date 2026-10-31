import random

import pytest
import torch
from ml_scratch.pure.mlp_mnist import ClassificationNetwork as ScalarNetwork
from ml_scratch.pure.mlp_mnist.experiment import run_experiment as scalar_experiment
from ml_scratch.pure.softmax_mnist.experiment import sparse_mnist
from ml_scratch.pytorch.mlp_mnist import (
    Activation,
    ClassificationNetwork,
    DenseLayer,
    Loss,
    NetworkTrainingConfig,
    byte_images,
)
from ml_scratch.pytorch.mlp_mnist import model as torch_model
from ml_scratch.pytorch.mlp_mnist.experiment import run_experiment

from ml_scratch.pure.mnist import FILES, mnist_directory


def close(actual, expected, rtol=0.0, atol=0.0):
    """Exact unless tolerances are given; accepts tensors, lists, and tuples."""
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )


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
    close(network.parameters(), scalar.parameters())
    close(network.gradient(data), scalar.gradient(sparse), rtol=1e-12, atol=1e-16)
    config = NetworkTrainingConfig(learning_rate=0.5, max_epochs=4, batch_size=16, seed=7)
    result, reference = network.fit(data, config), scalar.fit(sparse, config)
    close(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-12)
    close(network.parameters(), scalar.parameters(), rtol=1e-11, atol=1e-14)
    assert network.loss(data) == pytest.approx(scalar.loss(sparse), rel=1e-12)
    assert network.predictions(data).tolist() == scalar.predictions(sparse)
    assert network.confusion_matrix(data).tolist() == scalar.confusion_matrix(sparse)
    assert network.accuracy(data) == scalar.accuracy(sparse)


@pytest.mark.parametrize("layers", ARCHITECTURES.values(), ids=ARCHITECTURES.keys())
def test_manual_backpropagation_matches_independent_autograd(layers):
    images, labels = byte_problem(count=60)
    data = byte_images(images, labels, 23)
    network = ClassificationNetwork(layers, SOFTMAX, 9)
    network.fit(data, NetworkTrainingConfig(learning_rate=0.3, max_epochs=2, batch_size=8))
    before = network.parameters()
    close(network.gradient(data), network.autograd_gradient(data), rtol=1e-12, atol=1e-16)
    close(network.parameters(), before)
    assert network._flat.grad is None and not network._flat.requires_grad


def test_chunked_evaluation_matches_one_pass(monkeypatch):
    images, labels = byte_problem(count=50)
    data = byte_images(images, labels, 23)
    network = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 1)
    network.fit(data, NetworkTrainingConfig(max_epochs=2, batch_size=8))
    whole = (network.loss(data), network.predictions(data), network.gradient(data))
    monkeypatch.setattr(torch_model, "CHUNK", 7)
    assert network.loss(data) == pytest.approx(whole[0], rel=1e-14)
    assert torch.equal(network.predictions(data), whole[1])
    close(network.gradient(data), whole[2], rtol=1e-12, atol=1e-17)
    close(network.autograd_gradient(data), whole[2], rtol=1e-12, atol=1e-16)


def test_plain_examples_checkpoints_and_reproducibility(tmp_path):
    images, labels = byte_problem(count=40)
    data = byte_images(images, labels, 23)
    examples = [([byte / 255.0 for byte in image], label) for image, label in zip(images, labels)]
    network = ClassificationNetwork(ARCHITECTURES["tanh then sigmoid"], SOFTMAX, 3)
    assert network.loss(examples) == network.loss(data)
    config = NetworkTrainingConfig(learning_rate=0.3, max_epochs=3, batch_size=6, seed=4)
    twin = ClassificationNetwork(ARCHITECTURES["tanh then sigmoid"], SOFTMAX, 3)
    assert network.fit(data, config) == twin.fit(examples, config)
    assert torch.equal(network.parameters(), twin.parameters())
    path = tmp_path / "mlp.checkpoint"
    network.save(path)
    restored = ClassificationNetwork.load(path)
    assert type(restored) is ClassificationNetwork
    assert torch.equal(restored.parameters(), network.parameters())
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
        [([float("nan")] + [0.0] * 22, 1)],
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
    result = run_experiment(checkpoint=tmp_path / "torch.checkpoint", **options)
    reference = scalar_experiment(checkpoint=tmp_path / "scalar.checkpoint", **options)
    for ours, theirs in zip(result.candidates, reference.candidates, strict=True):
        assert ours.validation_accuracy == theirs.validation_accuracy
        assert ours.final_loss == pytest.approx(theirs.final_loss, rel=1e-11)
    assert result.selected.name == reference.selected.name
    close(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-11)
    assert result.validation_per_epoch == reference.validation_per_epoch
    assert result.mlp_confusion == reference.mlp_confusion
    assert result.linear_confusion == reference.linear_confusion
