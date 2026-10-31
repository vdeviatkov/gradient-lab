import random
from pathlib import Path

import pytest
from ml_scratch.pure.mlp_mnist import ClassificationNetwork
from ml_scratch.pure.mlp_mnist.experiment import CANDIDATES, run_experiment
from ml_scratch.pure.softmax_mnist.experiment import sparse_mnist

from ml_scratch.pure.mnist import FILES, load_mnist_directory, mnist_directory, write_idx

HAS_MNIST = all((mnist_directory() / name).is_file() for name in FILES)
# Written by the C++ experiment (./build/cpp_mlp_mnist) into the ignored data/ directory.
CPP_CHECKPOINT = Path("data/mlp_mnist_best.checkpoint")


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


def test_protocol_runs_end_to_end_and_restores_its_checkpoint(tmp_path):
    fake_mnist(tmp_path)
    checkpoint = tmp_path / "out" / "best.checkpoint"
    result = run_experiment(
        tmp_path,
        search_epochs=1,
        final_epochs=4,
        linear_epochs=2,
        validation_size=20,
        checkpoint=checkpoint,
        check=False,
    )
    assert result.split_sizes == (100, 20, 40) and result.search_epochs == 1
    assert [(c.hidden_units, c.activation, c.learning_rate) for c in result.candidates] == list(
        CANDIDATES
    )
    assert [c.parameter_count for c in result.candidates] == [16 * 64 + 64 + 650] * 2 + [
        16 * 128 + 128 + 1290
    ] * 2
    assert result.selected.validation_accuracy == max(
        c.validation_accuracy for c in result.candidates
    )
    assert len(result.loss_per_epoch) == len(result.validation_per_epoch) == 4
    # The checkpoint holds the first epoch with the best validation accuracy.
    assert result.best_epoch == result.validation_per_epoch.index(result.best_validation) + 1
    assert result.restored_validation == result.best_validation and result.checkpoint_matches
    restored = ClassificationNetwork.load(checkpoint)
    assert restored.parameter_count == result.mlp_parameter_count
    assert result.linear_parameter_count == 10 * 17
    for confusion, metrics in (
        (result.mlp_confusion, result.mlp),
        (result.linear_confusion, result.linear),
    ):
        assert sum(map(sum, confusion)) == 40
        assert metrics.accuracy == sum(confusion[k][k] for k in range(10)) / 40
    with pytest.raises(RuntimeError, match="criteria"):
        run_experiment(
            tmp_path,
            search_epochs=1,
            final_epochs=1,
            linear_epochs=1,
            validation_size=20,
            checkpoint=checkpoint,
        )


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST is not downloaded; see scripts/download_mnist.sh")
def test_real_mnist_subset_learns(tmp_path):
    result = run_experiment(
        max_samples=2000,
        search_epochs=1,
        final_epochs=3,
        linear_epochs=3,
        validation_size=500,
        checkpoint=tmp_path / "best.checkpoint",
        check=False,
    )
    assert result.split_sizes == (1500, 500, 2000)
    # Sanity floors for about 70 updates on 1500 images (measured: MLP 0.7590, linear 0.8000);
    # too short a run for the MLP to pull ahead, but far above the 0.1 of chance.
    assert result.mlp.accuracy > 0.7 and result.linear.accuracy > 0.7
    assert result.loss_per_epoch[-1] < result.loss_per_epoch[0]
    assert result.restored_validation == result.best_validation


@pytest.mark.skipif(
    not (HAS_MNIST and CPP_CHECKPOINT.is_file()),
    reason="needs MNIST and the checkpoint written by ./build/cpp_mlp_mnist",
)
def test_a_cpp_checkpoint_restores_the_cpp_network(tmp_path):
    network = ClassificationNetwork.load(CPP_CHECKPOINT)
    # The values are rewritten exactly as C++ wrote them; only the format version line may differ.
    network.save(tmp_path / "resaved.checkpoint")
    ours = (tmp_path / "resaved.checkpoint").read_text().split("\nparameters ")[1]
    assert ours == CPP_CHECKPOINT.read_text().split("\nparameters ")[1]
    _, test = load_mnist_directory(max_samples=1000)
    data = sparse_mnist(test.images, test.labels, 784)
    # A trained C++ MLP: the full test split gave 0.9814 on the recorded run.
    assert network.accuracy(data) > 0.95
