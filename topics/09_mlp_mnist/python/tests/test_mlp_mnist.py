import math
import random

import pytest
from ml_scratch.pure.backpropagation import FeedForwardNetwork
from ml_scratch.pure.mlp_mnist import (
    Activation,
    ClassificationNetwork,
    DenseLayer,
    Loss,
    NetworkTrainingConfig,
    sparse_dataset,
)

ID, S, T, R = Activation.IDENTITY, Activation.SIGMOID, Activation.TANH, Activation.RELU
SOFTMAX = Loss.SOFTMAX_CROSS_ENTROPY
ARCHITECTURES = {
    "softmax only": (DenseLayer(9, 4),),
    "one ReLU layer": (DenseLayer(9, 6, R), DenseLayer(6, 4)),
    "tanh then sigmoid": (DenseLayer(9, 5, T), DenseLayer(5, 3, S), DenseLayer(3, 4)),
}


def sparse_problem(count=40, width=9, classes=4, seed=3):
    """Mostly-zero rows where class k brightens feature k, so the sparse path is exercised."""
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        label = rng.randrange(classes)
        row = [rng.choice((0.0, 0.0, rng.uniform(0.0, 1.0))) for _ in range(width)]
        row[label] = 1.0
        examples.append((row, label))
    return examples


def one_hot(examples, classes=4):
    return [(x, [1.0 if k == label else 0.0 for k in range(classes)]) for x, label in examples]


@pytest.mark.parametrize("layers", ARCHITECTURES.values(), ids=ARCHITECTURES.keys())
def test_class_index_methods_match_the_topic_07_network_on_one_hot_targets(layers):
    examples = sparse_problem()
    network = ClassificationNetwork(layers, SOFTMAX, 11)
    reference = FeedForwardNetwork(layers, SOFTMAX, 11)
    assert network.parameters() == reference.parameters()
    assert network.loss(examples) == pytest.approx(reference.loss(one_hot(examples)), rel=1e-14)
    assert network.gradient(examples) == pytest.approx(
        reference.gradient(one_hot(examples)), rel=1e-12, abs=1e-16
    )
    assert network.accuracy(examples) == reference.accuracy(one_hot(examples))
    assert network.predictions(examples) == [reference.predict_class(x) for x, _ in examples]

    config = NetworkTrainingConfig(learning_rate=0.4, max_epochs=6, batch_size=7, seed=2)
    result = network.fit(examples, config)
    expected = reference.fit(one_hot(examples), config)
    assert result.epochs == expected.epochs and result.converged == expected.converged
    assert result.loss_per_epoch == pytest.approx(expected.loss_per_epoch, rel=1e-12)
    assert network.parameters() == pytest.approx(reference.parameters(), rel=1e-11, abs=1e-14)


def test_sparse_rows_skip_zeros_exactly():
    examples = sparse_problem()
    data = sparse_dataset(examples)
    assert sum(map(len, data.indices)) < 0.6 * len(examples) * 9
    network = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 4)
    assert network.loss(data) == network.loss(examples)
    assert network.gradient(data) == network.gradient(examples)
    # A row of zeros reads no first-layer weight, and a batch of them leaves those weights alone.
    zeros = [([0.0] * 9, 1), ([0.0] * 9, 2)]
    before = network.parameters()
    network.fit(zeros, NetworkTrainingConfig(max_epochs=1))
    after = network.parameters()
    assert after[: 6 * 9] == before[: 6 * 9] and after != before


def test_learning_confusion_matrix_and_reproducibility():
    examples = sparse_problem(count=80)
    config = NetworkTrainingConfig(learning_rate=0.5, max_epochs=40, batch_size=8, seed=5)
    first = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 1)
    second = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 1)
    result = first.fit(examples, config)
    assert result == second.fit(examples, config) and first.parameters() == second.parameters()
    assert result.loss_per_epoch[-1] < 0.3 * result.loss_per_epoch[0]
    assert first.accuracy(examples) > 0.9
    confusion = first.confusion_matrix(examples)
    assert sum(map(sum, confusion)) == 80
    assert sum(confusion[k][k] for k in range(4)) == round(first.accuracy(examples) * 80)
    converged = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 1).fit(
        examples, NetworkTrainingConfig(learning_rate=0.5, max_epochs=500, target_loss=0.5)
    )
    assert converged.converged and converged.loss_per_epoch[-1] <= 0.5


def test_checkpoints_restore_this_class_exactly(tmp_path):
    examples = sparse_problem()
    network = ClassificationNetwork(ARCHITECTURES["tanh then sigmoid"], SOFTMAX, 8)
    network.fit(examples, NetworkTrainingConfig(learning_rate=0.3, max_epochs=3, batch_size=5))
    path = tmp_path / "mlp.checkpoint"
    network.save(path)
    restored = ClassificationNetwork.load(path)
    assert type(restored) is ClassificationNetwork
    assert restored.parameters() == network.parameters()
    assert restored.loss(examples) == network.loss(examples)


def test_divergence_and_invalid_input():
    examples = sparse_problem()
    network = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 2)
    with pytest.raises(ArithmeticError):
        network.fit(examples, NetworkTrainingConfig(learning_rate=1e300, max_epochs=3))
    with pytest.raises(ValueError):
        ClassificationNetwork(ARCHITECTURES["one ReLU layer"], Loss.MEAN_SQUARED_ERROR)
    network = ClassificationNetwork(ARCHITECTURES["one ReLU layer"], SOFTMAX, 2)
    for data in (
        [],
        [([0.0] * 9, 4)],
        [([0.0] * 9, -1)],
        [([0.0] * 8, 1)],
        [([math.nan] + [0.0] * 8, 1)],
        sparse_dataset([([1.0] * 5, 0)]),
    ):
        with pytest.raises(ValueError):
            network.loss(data)
