# The imports below follow the skip guard on purpose: Triton exists only on Linux.
# ruff: noqa: E402
import math
import os
import random

import pytest

pytest.importorskip("triton", reason="Triton is not installed (Linux only)")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs a CUDA device or TRITON_INTERPRET=1", allow_module_level=True)

from ml_scratch.pure.softmax_mnist import SoftmaxRegression as ScalarModel
from ml_scratch.pure.softmax_mnist.experiment import (
    run_experiment as scalar_experiment,
)
from ml_scratch.pure.softmax_mnist.experiment import sparse_mnist
from ml_scratch.triton.softmax_mnist import (
    SoftmaxRegression,
    SoftmaxTrainingConfig,
    byte_images,
)
from ml_scratch.triton.softmax_mnist import model as triton_model
from ml_scratch.triton.softmax_mnist.experiment import run_experiment

from ml_scratch.pure.mnist import FILES, mnist_directory

HAS_MNIST = all((mnist_directory() / name).is_file() for name in FILES)


def close(actual, expected, rtol=0.0, atol=0.0):
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64).cpu(),
        torch.as_tensor(expected, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )


def byte_problem(count=150, width=23, classes=4, seed=2):
    """Class k brightens feature k. 23 features, 150 rows, and batch 13 all leave partial tiles."""
    rng = random.Random(seed)
    labels = [rng.randrange(classes) for _ in range(count)]
    images = []
    for label in labels:
        image = [rng.choice((0, 0, rng.randrange(256))) for _ in range(width)]
        image[label] = 255
        images.append(bytes(image))
    return images, labels


@pytest.mark.parametrize("classes,width", [(4, 23), (10, 70), (17, 33)])
@pytest.mark.parametrize("l2", [0.0, 1e-3])
def test_kernels_follow_the_scalar_trajectory(classes, width, l2):
    # 17 classes pad to a 32-wide class tile; 10 and 4 pad to 16.
    images, labels = byte_problem(width=width, classes=classes)
    config = SoftmaxTrainingConfig(
        learning_rate=0.5, max_epochs=3, batch_size=13, seed=7, l2_regularization=l2
    )
    model, scalar = SoftmaxRegression(width, classes), ScalarModel(width, classes)
    data, sparse = byte_images(images, labels, width), sparse_mnist(images, labels, width)
    result, reference = model.fit(data, config), scalar.fit(sparse, config)
    close(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-13)
    close(model.parameters(), scalar.parameters(), rtol=1e-12, atol=1e-15)
    close(model.gradient(data, l2), scalar.gradient(sparse, l2), atol=1e-15)
    close(model.predictions(data), scalar.predictions(sparse))
    close(model.confusion_matrix(data), scalar.confusion_matrix(sparse))
    row = [byte / 255.0 for byte in images[0]]
    close(model.predict_probabilities(row), scalar.predict_probabilities(row), rtol=1e-13)
    assert model.predict(row) == scalar.predict(row)


def test_byte_and_float_paths_agree_and_runs_are_bit_reproducible():
    images, labels = byte_problem()
    config = SoftmaxTrainingConfig(learning_rate=0.5, max_epochs=2, batch_size=16, seed=3)
    by_bytes = SoftmaxRegression(23, 4)
    by_floats = SoftmaxRegression(23, 4)
    again = SoftmaxRegression(23, 4)
    dense = [
        ([b / 255.0 for b in image], label) for image, label in zip(images, labels, strict=True)
    ]
    first = by_bytes.fit(byte_images(images, labels, 23), config)
    assert first == by_floats.fit(dense, config)
    assert torch.equal(by_bytes.parameters(), by_floats.parameters())
    # Each weight is reduced by one program in a fixed order: no atomics, no run-to-run drift.
    assert again.fit(byte_images(images, labels, 23), config) == first
    assert torch.equal(again.parameters(), by_bytes.parameters())


def test_evaluation_chunks_and_full_batch_gradient(monkeypatch):
    images, labels = byte_problem(count=60)
    data = byte_images(images, labels, 23)
    model = SoftmaxRegression(23, 4)
    model.fit(data, SoftmaxTrainingConfig(max_epochs=2, batch_size=0))
    whole = (model.cross_entropy(data, 1e-3), model.predictions(data))
    monkeypatch.setattr(triton_model, "CHUNK", 7)
    assert model.cross_entropy(data, 1e-3) == pytest.approx(whole[0], rel=1e-14)
    assert torch.equal(model.predictions(data), whole[1])


def test_scalar_kernel_arguments_keep_float64_precision():
    # Triton would pass a Python float as fp32; 1 / batch and the hyperparameters must not be.
    model = SoftmaxRegression(1, 2)
    data = [([1.0], 0)] * 3
    model.fit(data, SoftmaxTrainingConfig(learning_rate=0.1 / 3, max_epochs=1, batch_size=0))
    # One step from zero: dL/dw_0 = 0.5 - 1, so w_0 = -(0.1 / 3) * (-0.5) exactly in float64.
    assert float(model.weights[0, 0]) == (0.1 / 3) * 0.5


# Under TRITON_INTERPRET=1 the deliberate divergence overflows in NumPy, which warns.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_saturation_divergence_and_invalid_inputs():
    model = SoftmaxRegression(1, 3)
    model.set_parameters([1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    assert model.cross_entropy([([1.0], 1)]) == pytest.approx(2000.0)
    with pytest.raises(ArithmeticError):
        SoftmaxRegression(1, 2).fit(
            [([1e200], 0), ([-1e200], 1)], SoftmaxTrainingConfig(learning_rate=1e200, max_epochs=5)
        )
    for build in (lambda: SoftmaxRegression(0, 2), lambda: SoftmaxRegression(2, 129)):
        with pytest.raises(ValueError):
            build()
    model = SoftmaxRegression(2, 2)
    for data in ([], [([1.0], 0)], [([1.0, math.inf], 0)], [([1.0, 0.0], 2)], [([1.0, 0.0], 0.5)]):
        with pytest.raises(ValueError):
            model.cross_entropy(data)
    with pytest.raises(ValueError):
        model.set_parameters([0.0] * 5)


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST is not downloaded; see scripts/download_mnist.sh")
def test_protocol_on_an_mnist_subset_matches_scalar():
    options = dict(
        max_samples=2000, sweep_epochs=1, final_epochs=2, validation_size=400, check=False
    )
    result, reference = run_experiment(**options), scalar_experiment(**options)
    close(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-12)
    assert result.confusion == reference.confusion and result.test == reference.test


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST is not downloaded; see scripts/download_mnist.sh")
def test_full_mnist_experiment_meets_the_criteria():
    result = run_experiment()  # Raises unless every recorded criterion holds.
    assert (result.selected.learning_rate, result.selected.l2_regularization) == (0.1, 0.0)
    assert round(result.test.accuracy, 4) == 0.9231 and round(result.test.macro_f1, 4) == 0.9220
    assert result.most_frequent_confusion == (9, 4, 40)
