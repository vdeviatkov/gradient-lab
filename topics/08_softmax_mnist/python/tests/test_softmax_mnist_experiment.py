import random

import pytest
from ml_scratch.pure.softmax_mnist.experiment import CANDIDATES, run_experiment, sparse_mnist

from ml_scratch.pure.mnist import FILES, load_mnist_directory, mnist_directory, write_idx

HAS_MNIST = all((mnist_directory() / name).is_file() for name in FILES)


def fake_mnist(directory, train=120, test=40, side=4, seed=0):
    """A tiny IDX directory where digit k lights pixel k amid noise; enough for the protocol."""
    rng = random.Random(seed)
    for prefix, count in (("train", train), ("t10k", test)):
        labels = [index % 10 for index in range(count)]
        images = bytearray()
        for label in labels:
            image = [rng.choice((0, 0, 0, rng.randrange(1, 80))) for _ in range(side * side)]
            image[label] = 255
            images.extend(image)
        write_idx(directory / f"{prefix}-images-idx3-ubyte", (count, side, side), images)
        write_idx(directory / f"{prefix}-labels-idx1-ubyte", (count,), labels)


def test_byte_scaling_matches_the_cpp_loader():
    data = sparse_mnist([bytes([0, 255, 3, 0])], [7], 4)
    assert data.indices == ((1, 2),) and data.values == ((1.0, 3 / 255.0),)
    assert data.labels == (7,) and data.feature_count == 4


def test_protocol_runs_end_to_end_on_a_synthetic_idx_directory(tmp_path):
    fake_mnist(tmp_path)
    result = run_experiment(
        tmp_path, sweep_epochs=2, final_epochs=4, validation_size=20, check=False
    )
    assert result.split_sizes == (100, 20, 40)
    assert result.validation_counts == (2,) * 10
    assert len(result.candidates) == len(CANDIDATES) and len(result.loss_per_epoch) == 4
    assert result.selected.validation_accuracy == max(
        c.validation_accuracy for c in result.candidates
    )
    assert sum(map(sum, result.confusion)) == 40
    assert result.test.accuracy == pytest.approx(
        sum(result.confusion[k][k] for k in range(10)) / 40
    )
    assert result.parameter_count == 10 * 17
    assert result.test.accuracy > result.baseline_accuracy
    with pytest.raises(RuntimeError, match="criteria"):
        run_experiment(tmp_path, sweep_epochs=1, final_epochs=1, validation_size=20, max_samples=30)


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST is not downloaded; see scripts/download_mnist.sh")
def test_real_mnist_subset_learns():
    train, test = load_mnist_directory(max_samples=3000)
    assert (train.rows, train.columns) == (28, 28) and len(test.labels) == 3000
    result = run_experiment(
        max_samples=3000, sweep_epochs=1, final_epochs=3, validation_size=500, check=False
    )
    assert result.split_sizes == (2500, 500, 3000)
    # A sanity floor for 2500 images and three epochs (measured 0.79), far above chance.
    assert result.test.accuracy > 0.7 > 6 * result.baseline_accuracy
    assert result.loss_per_epoch[-1] < result.loss_per_epoch[0]
