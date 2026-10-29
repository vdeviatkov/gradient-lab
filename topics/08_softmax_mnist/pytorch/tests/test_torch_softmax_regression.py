import math
import random
from dataclasses import astuple

import pytest
import torch
from ml_scratch.pure.softmax_mnist import SoftmaxRegression as ScalarModel
from ml_scratch.pure.softmax_mnist.experiment import run_experiment as scalar_experiment
from ml_scratch.pure.softmax_mnist.experiment import sparse_mnist
from ml_scratch.pytorch.softmax_mnist import SoftmaxRegression, SoftmaxTrainingConfig, byte_images
from ml_scratch.pytorch.softmax_mnist import model as torch_model
from ml_scratch.pytorch.softmax_mnist.experiment import run_experiment

from ml_scratch.pure.mnist import FILES, load_mnist_directory, mnist_directory


def close(actual, expected, rtol=0.0, atol=0.0):
    """Exact unless tolerances are given; accepts tensors, lists, and tuples."""
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )


HAS_MNIST = all((mnist_directory() / name).is_file() for name in FILES)


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


@pytest.mark.parametrize("l2", [0.0, 1e-3])
def test_training_trajectory_and_evaluation_match_scalar(l2):
    images, labels = byte_problem()
    config = SoftmaxTrainingConfig(
        learning_rate=0.5, max_epochs=4, batch_size=16, seed=7, l2_regularization=l2
    )
    model, scalar = SoftmaxRegression(23, 4), ScalarModel(23, 4)
    data, sparse = byte_images(images, labels, 23), sparse_mnist(images, labels, 23)
    result, reference = model.fit(data, config), scalar.fit(sparse, config)
    close(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-13)
    close(model.parameters(), scalar.parameters(), rtol=1e-12, atol=1e-15)
    close(model.gradient(data, l2), scalar.gradient(sparse, l2), atol=1e-15)
    close(model.predictions(data), scalar.predictions(sparse))
    close(model.confusion_matrix(data), scalar.confusion_matrix(sparse))
    assert model.accuracy(data) == scalar.accuracy(sparse)
    row = data.features(0)
    close(model.predict_probabilities(row), scalar.predict_probabilities(row), rtol=1e-13)


def test_chunked_evaluation_matches_one_pass(monkeypatch):
    images, labels = byte_problem(count=50)
    data = byte_images(images, labels, 23)
    model = SoftmaxRegression(23, 4)
    model.fit(data, SoftmaxTrainingConfig(max_epochs=2, batch_size=8))
    whole = (model.cross_entropy(data, 1e-3), model.predictions(data), model.gradient(data))
    monkeypatch.setattr(torch_model, "CHUNK", 7)
    assert model.cross_entropy(data, 1e-3) == pytest.approx(whole[0], rel=1e-14)
    close(model.predictions(data), whole[1])
    close(model.gradient(data), whole[2], atol=1e-16)


def test_bytes_stay_bytes_and_scale_exactly():
    data = byte_images([bytes([0, 1, 255])], [0], 3)
    assert data.pixels.dtype == torch.uint8 and data.pixels.numel() == 3
    assert data.features(0).tolist() == [0.0, 1 / 255.0, 1.0]


def test_dense_examples_parameters_and_saturation():
    model = SoftmaxRegression(2, 3)
    assert model.cross_entropy([([1.0, 2.0], 0)]) == pytest.approx(math.log(3))
    model.set_parameters([1000.0, 0.0, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    close(model.weights, [[1000, 0], [-1000, 0], [0, 0]])
    assert model.cross_entropy([([1.0, 0.0], 1)]) == pytest.approx(2000.0)
    close(model.predict_probabilities([1.0, 0.0]), [1, 0, 0], atol=1e-300)
    copy = model.parameters()
    copy[0] = 5.0
    assert float(model.weights[0, 0]) == 1000.0


def test_divergence_and_invalid_inputs():
    with pytest.raises(ArithmeticError):
        SoftmaxRegression(1, 2).fit(
            [([1e200], 0), ([-1e200], 1)], SoftmaxTrainingConfig(learning_rate=1e200, max_epochs=5)
        )
    model = SoftmaxRegression(2, 2)
    for data in ([], [([1.0], 0)], [([1.0, math.inf], 0)], [([1.0, 0.0], 2)], [([1.0, 0.0], 0.5)]):
        with pytest.raises(ValueError):
            model.cross_entropy(data)
    for bad in (lambda: SoftmaxRegression(0, 2), lambda: byte_images([b"\x00"], [0, 1], 1)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        model.set_parameters([0.0, math.nan, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        model.cross_entropy(byte_images([b"\x00\x00\x00"], [0], 3))


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST is not downloaded; see scripts/download_mnist.sh")
def test_protocol_on_an_mnist_subset_matches_scalar():
    options = dict(
        max_samples=2000, sweep_epochs=1, final_epochs=2, validation_size=400, check=False
    )
    result, reference = run_experiment(**options), scalar_experiment(**options)
    assert astuple(result.selected) == pytest.approx(astuple(reference.selected))
    close(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-12)
    assert result.confusion == reference.confusion
    assert result.test == reference.test  # Same metric code on identical confusion matrices.


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST is not downloaded; see scripts/download_mnist.sh")
def test_full_mnist_experiment_meets_the_criteria():
    result = run_experiment()  # Raises unless every recorded criterion holds.
    assert result.split_sizes == (54000, 6000, 10000) and result.parameter_count == 7850
    # Recorded scalar-Python values; the shuffle stream is shared, so these are reproduced.
    assert (result.selected.learning_rate, result.selected.l2_regularization) == (0.1, 0.0)
    assert round(result.test.accuracy, 4) == 0.9231 and round(result.test.macro_f1, 4) == 0.9220
    assert result.most_frequent_confusion == (9, 4, 40)
    assert result.baseline_accuracy == pytest.approx(0.1135)


def test_loaded_subset_is_byte_identical_to_files():
    if not HAS_MNIST:
        pytest.skip("MNIST is not downloaded")
    train, _ = load_mnist_directory(max_samples=5)
    data = byte_images(train.images, train.labels, 784)
    assert bytes(data.pixels.reshape(-1).tolist()) == b"".join(train.images)


@pytest.mark.parametrize("l2", [0.0, 1e-3])
def test_manual_gradient_matches_independent_autograd(l2):
    images, labels = byte_problem()
    data = byte_images(images, labels, 23)
    model = SoftmaxRegression(23, 4)
    model.fit(data, SoftmaxTrainingConfig(learning_rate=0.5, max_epochs=2, batch_size=16))
    before = model.parameters()
    torch.testing.assert_close(
        model.gradient(data, l2), model.autograd_gradient(data, l2), rtol=1e-12, atol=1e-15
    )
    assert torch.equal(model.parameters(), before) and model._flat.grad is None
    assert model.gradient(data).dtype == torch.float64
