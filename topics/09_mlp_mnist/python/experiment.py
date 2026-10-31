"""Put one hidden layer in front of the softmax classifier and measure what it buys on MNIST."""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ml_scratch.pure.backpropagation.model import (
    Activation,
    DenseLayer,
    Loss,
    NetworkTrainingConfig,
)
from ml_scratch.pure.softmax_mnist.experiment import sparse_mnist
from ml_scratch.pure.softmax_mnist.metrics import MulticlassMetrics, multiclass_metrics
from ml_scratch.pure.softmax_mnist.model import SoftmaxRegression, SoftmaxTrainingConfig

from ml_scratch.pure.mnist import load_mnist_directory

from .model import ClassificationNetwork

# Identical to experiment 08, so the two models are compared under the same conditions.
SEED = 20260911
CLASS_COUNT = 10
VALIDATION_SIZE = 6000
BATCH_SIZE = 64
SEARCH_EPOCHS = 5
FINAL_EPOCHS = 25
# The softmax baseline is retrained in this run with the C++ experiment 08 selection.
LINEAR_LEARNING_RATE = 0.5
LINEAR_EPOCHS = 30
CANDIDATES = (
    (64, Activation.RELU, 0.10),
    (64, Activation.TANH, 0.10),
    (128, Activation.RELU, 0.10),
    (128, Activation.RELU, 0.50),
)
# The digit pairs whose confusions limited the linear model in experiment 08.
WATCHED_CONFUSIONS = ((5, 3), (4, 9), (7, 9), (8, 3))
REPORTED_EPOCHS = (1, 5, 10, 15, 20, 25)
ACTIVATION_NAMES = {Activation.RELU: "ReLU", Activation.TANH: "tanh"}


def default_checkpoint(backend):
    """Each backend checkpoints to its own file inside the ignored data/ directory."""
    return Path("data") / f"mlp_mnist_best.{backend}.checkpoint"


def build_layers(hidden_units, activation, feature_count):
    return (
        DenseLayer(feature_count, hidden_units, activation),
        DenseLayer(hidden_units, CLASS_COUNT),
    )


@dataclass(frozen=True)
class Candidate:
    hidden_units: int
    activation: Activation
    learning_rate: float
    parameter_count: int
    final_loss: float
    validation_accuracy: float

    @property
    def name(self):
        units = f"{self.hidden_units} {ACTIVATION_NAMES[self.activation]},"
        return f"{units:<10}lr {self.learning_rate:.2f}"


@dataclass(frozen=True)
class ExperimentResult:
    split_sizes: tuple[int, int, int]
    search_epochs: int
    candidates: tuple[Candidate, ...]
    selected: Candidate
    loss_per_epoch: tuple[float, ...]
    validation_per_epoch: tuple[float, ...]
    best_epoch: int
    checkpoint: Path
    restored_validation: float
    checkpoint_matches: bool
    training_seconds: float
    mlp_parameter_count: int
    linear_parameter_count: int
    mlp: MulticlassMetrics
    linear: MulticlassMetrics
    mlp_confusion: tuple[tuple[int, ...], ...]
    linear_confusion: tuple[tuple[int, ...], ...]

    @property
    def best_validation(self):
        return self.validation_per_epoch[self.best_epoch - 1]

    @property
    def error_reduction(self):
        return 1.0 - (1.0 - self.mlp.accuracy) / (1.0 - self.linear.accuracy)


def _values(parameters):
    """A plain list of floats from a list, NumPy array, or tensor."""
    return parameters.tolist() if hasattr(parameters, "tolist") else list(parameters)


def _confusion(matrix):
    return tuple(tuple(int(count) for count in row) for row in matrix)


def run_with(
    network_type,
    linear_type,
    prepare,
    directory=None,
    seed=SEED,
    search_epochs=SEARCH_EPOCHS,
    final_epochs=FINAL_EPOCHS,
    linear_epochs=LINEAR_EPOCHS,
    max_samples=0,
    validation_size=VALIDATION_SIZE,
    checkpoint=None,
    check=True,
):
    """Run the C++ protocol with any backend's network, softmax model, and dataset builder.

    `prepare(images, labels, features)` builds the dataset both models read. `max_samples`,
    `validation_size`, and the epoch counts shrink the run for tests; `check` enforces the C++
    success criteria.
    """
    checkpoint = Path(checkpoint or default_checkpoint("python"))
    train_file, test_file = load_mnist_directory(directory, max_samples)
    if len(train_file.labels) <= validation_size:
        raise ValueError("training split is too small to hold out a validation set")
    features = train_file.rows * train_file.columns
    # The last images of the official training file validate; the test file is used once.
    cut = len(train_file.labels) - validation_size
    train = prepare(train_file.images[:cut], train_file.labels[:cut], features)
    validation = prepare(train_file.images[cut:], train_file.labels[cut:], features)
    test = prepare(test_file.images, test_file.labels, features)

    def config(learning_rate, epochs, shuffle_seed):
        return NetworkTrainingConfig(
            learning_rate=learning_rate,
            max_epochs=epochs,
            batch_size=BATCH_SIZE,
            seed=shuffle_seed,
        )

    candidates = []
    for hidden_units, activation, learning_rate in CANDIDATES:
        layers = build_layers(hidden_units, activation, features)
        network = network_type(layers, Loss.SOFTMAX_CROSS_ENTROPY, seed)
        result = network.fit(train, config(learning_rate, search_epochs, seed))
        candidates.append(
            Candidate(
                hidden_units,
                activation,
                learning_rate,
                network.parameter_count,
                result.loss_per_epoch[-1],
                network.accuracy(validation),
            )
        )
    # max() keeps the first maximum, matching the C++ strict-improvement scan.
    selected = max(candidates, key=lambda c: c.validation_accuracy)

    # The epoch loop is driven here rather than inside fit, so validation accuracy is measured
    # after every epoch and the best parameters checkpointed. Each epoch advances the shuffle seed.
    layers = build_layers(selected.hidden_units, selected.activation, features)
    network = network_type(layers, Loss.SOFTMAX_CROSS_ENTROPY, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    losses, accuracies, best_epoch = [], [], 0
    start = time.perf_counter()
    for epoch in range(1, final_epochs + 1):
        result = network.fit(train, config(selected.learning_rate, 1, seed + epoch))
        losses.append(result.loss_per_epoch[-1])
        accuracies.append(network.accuracy(validation))
        if best_epoch == 0 or accuracies[-1] > accuracies[best_epoch - 1]:
            best_epoch = epoch
            network.save(checkpoint)
    seconds = time.perf_counter() - start

    # Restoring the checkpoint is how the evaluated model is obtained, so the round trip is
    # exercised for real rather than only in a unit test.
    restored = network_type.load(checkpoint)
    checkpoint_matches = best_epoch != final_epochs or _values(restored.parameters()) == _values(
        network.parameters()
    )
    restored_validation = restored.accuracy(validation)

    linear = linear_type(features, CLASS_COUNT)
    linear.fit(
        train,
        SoftmaxTrainingConfig(
            learning_rate=LINEAR_LEARNING_RATE,
            max_epochs=linear_epochs,
            batch_size=BATCH_SIZE,
            seed=seed,
        ),
    )

    # Everything above used train and validation only. The test split is evaluated here, once.
    mlp_confusion = _confusion(restored.confusion_matrix(test))
    linear_confusion = _confusion(linear.confusion_matrix(test))
    outcome = ExperimentResult(
        (cut, validation_size, len(test_file.labels)),
        search_epochs,
        tuple(candidates),
        selected,
        tuple(losses),
        tuple(accuracies),
        best_epoch,
        checkpoint,
        restored_validation,
        checkpoint_matches,
        seconds,
        restored.parameter_count,
        linear.parameter_count,
        multiclass_metrics(mlp_confusion),
        multiclass_metrics(linear_confusion),
        mlp_confusion,
        linear_confusion,
    )
    if check and not (
        outcome.mlp.accuracy > outcome.linear.accuracy
        and restored_validation == outcome.best_validation
        and checkpoint_matches
        and losses[-1] < losses[0]
    ):
        raise RuntimeError(
            f"experiment failed its criteria: MLP accuracy={outcome.mlp.accuracy:.4f}, "
            f"linear accuracy={outcome.linear.accuracy:.4f}, "
            f"restored validation={restored_validation:.4f}"
        )
    return outcome


def run_experiment(directory=None, seed=SEED, **options):
    options.setdefault("checkpoint", default_checkpoint("python"))
    return run_with(
        ClassificationNetwork, SoftmaxRegression, sparse_mnist, directory, seed, **options
    )


def print_result(title, seed, result):
    train, validation, test = result.split_sizes
    print(f"{title} MLP on MNIST (seed={seed})")
    print(
        "same data, preprocessing, splits and seed as experiment 08, so the softmax baseline "
        "below is measured under identical conditions"
    )
    print(f"splits: train={train}, validation={validation}, test={test}\n")
    print(
        f"architecture search on the validation split ({result.search_epochs} epochs, "
        f"batch={BATCH_SIZE})"
    )
    for c in result.candidates:
        print(
            f"  {c.name}: parameters={c.parameter_count:6}, train loss={c.final_loss:.4f}, "
            f"validation accuracy={c.validation_accuracy:.4f}"
        )
    print(f"  selected {result.selected.name} by validation accuracy\n")
    epochs = len(result.loss_per_epoch)
    print(f"training the selected network for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        if epoch in REPORTED_EPOCHS or epoch == result.best_epoch:
            print(
                f"  epoch {epoch:2}: training loss={result.loss_per_epoch[epoch - 1]:.4f}, "
                f"validation accuracy={result.validation_per_epoch[epoch - 1]:.4f}"
            )
    print(
        f"  best validation accuracy {result.best_validation:.4f} at epoch {result.best_epoch}, "
        f"checkpointed to {result.checkpoint}"
    )
    print(
        f"  wall-clock training time={result.training_seconds:.2f} s "
        f"({result.training_seconds / epochs:.2f} s per epoch)\n"
    )
    print("held-out results (test split evaluated once, from the restored checkpoint)")
    for name, parameters, m in (
        ("softmax regression", result.linear_parameter_count, result.linear),
        ("MLP", result.mlp_parameter_count, result.mlp),
    ):
        print(
            f"  {name:<18} ({parameters:6} params): test accuracy={m.accuracy:.4f}, "
            f"macro F1={m.macro_f1:.4f}"
        )
    print(
        f"  restored checkpoint validation accuracy={result.restored_validation:.4f} "
        f"(best during training {result.best_validation:.4f})"
    )
    print(
        f"  error rate fell from {1 - result.linear.accuracy:.4f} to "
        f"{1 - result.mlp.accuracy:.4f}, a {100 * result.error_reduction:.1f}% reduction\n"
    )
    print("per-digit test F1\n  digit   linear      MLP")
    for digit in range(CLASS_COUNT):
        print(
            f"  {digit:5}{result.linear.per_class_f1[digit]:9.4f}"
            f"{result.mlp.per_class_f1[digit]:9.4f}"
        )
    print("\nMLP test confusion matrix (rows actual, columns predicted)")
    for actual, row in enumerate(result.mlp_confusion):
        print(f"  {actual}" + "".join(f"{count:6}" for count in row))
    print("\nconfusions that limited the linear model")
    for actual, predicted in WATCHED_CONFUSIONS:
        print(
            f"  {actual} predicted as {predicted}: "
            f"linear {result.linear_confusion[actual][predicted]}, "
            f"MLP {result.mlp_confusion[actual][predicted]}"
        )


def main(run=run_experiment, title="Python", backend="python"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data", default=None, help="MNIST directory (default data/mnist)")
    parser.add_argument("--checkpoint", default=None, help=f"default {default_checkpoint(backend)}")
    args = parser.parse_args()
    try:
        result = run(
            args.data, args.seed, checkpoint=args.checkpoint or default_checkpoint(backend)
        )
    except FileNotFoundError as error:
        sys.exit(str(error))
    print_result(title, args.seed, result)


if __name__ == "__main__":
    main()
