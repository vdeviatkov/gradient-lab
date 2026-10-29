"""Select a learning rate and L2 on validation, retrain, then evaluate the MNIST test split once."""

import argparse
import sys
import time
from dataclasses import dataclass

from ml_scratch.pure.mnist import load_mnist_directory

from .metrics import MulticlassMetrics, multiclass_metrics
from .model import SoftmaxRegression, SoftmaxTrainingConfig, SparseDataset

SEED = 20260911
CLASS_COUNT = 10
VALIDATION_SIZE = 6000
SWEEP_EPOCHS = 8
FINAL_EPOCHS = 30
BATCH_SIZE = 64
CANDIDATES = ((0.10, 0.0), (0.10, 1e-4), (0.50, 0.0), (0.50, 1e-4), (0.50, 1e-3), (1.00, 1e-4))
REPORTED_EPOCHS = (1, 5, 10, 20, 30)


def sparse_mnist(images, labels, feature_count):
    """Each pixel byte b becomes b / 255.0, as in C++; zeros are dropped."""
    indices, values = [], []
    for image in images:
        kept = tuple(i for i, byte in enumerate(image) if byte)
        indices.append(kept)
        values.append(tuple(image[i] / 255.0 for i in kept))
    return SparseDataset(tuple(indices), tuple(values), tuple(labels), feature_count)


@dataclass(frozen=True)
class Candidate:
    learning_rate: float
    l2_regularization: float
    final_loss: float
    train_accuracy: float
    validation_accuracy: float


@dataclass(frozen=True)
class ExperimentResult:
    split_sizes: tuple[int, int, int]
    validation_counts: tuple[int, ...]
    candidates: tuple[Candidate, ...]
    selected: Candidate
    loss_per_epoch: tuple[float, ...]
    training_seconds: float
    parameter_count: int
    train_accuracy: float
    validation_accuracy: float
    test: MulticlassMetrics
    confusion: tuple[tuple[int, ...], ...]
    majority_class: int
    baseline_accuracy: float
    most_frequent_confusion: tuple[int, int, int]  # actual, predicted, count


def run_with(
    model_type,
    prepare,
    directory=None,
    seed=SEED,
    sweep_epochs=SWEEP_EPOCHS,
    final_epochs=FINAL_EPOCHS,
    max_samples=0,
    validation_size=VALIDATION_SIZE,
    check=True,
):
    """Run the protocol with any backend; `prepare(images, labels, features)` builds its dataset.

    `max_samples` and `validation_size` shrink the run for tests; `check` enforces the recorded
    success criteria, which only make sense on full MNIST.
    """
    train_file, test_file = load_mnist_directory(directory, max_samples)
    if len(train_file.labels) <= validation_size:
        raise ValueError("training split is too small to hold out a validation set")
    features = train_file.rows * train_file.columns
    # The last images of the official training file validate; the test file is used once.
    cut = len(train_file.labels) - validation_size
    train = prepare(train_file.images[:cut], train_file.labels[:cut], features)
    validation = prepare(train_file.images[cut:], train_file.labels[cut:], features)
    test = prepare(test_file.images, test_file.labels, features)
    validation_counts = tuple(train_file.labels[cut:].count(k) for k in range(CLASS_COUNT))

    def config(learning_rate, l2, epochs):
        return SoftmaxTrainingConfig(
            learning_rate=learning_rate,
            max_epochs=epochs,
            batch_size=BATCH_SIZE,
            seed=seed,
            l2_regularization=l2,
        )

    candidates = []
    for learning_rate, l2 in CANDIDATES:
        model = model_type(features, CLASS_COUNT)
        result = model.fit(train, config(learning_rate, l2, sweep_epochs))
        candidates.append(
            Candidate(
                learning_rate,
                l2,
                result.loss_per_epoch[-1],
                model.accuracy(train),
                model.accuracy(validation),
            )
        )
    # max() keeps the first maximum, matching the C++ strict-improvement scan.
    selected = max(candidates, key=lambda c: c.validation_accuracy)

    model = model_type(features, CLASS_COUNT)
    start = time.perf_counter()
    result = model.fit(
        train, config(selected.learning_rate, selected.l2_regularization, final_epochs)
    )
    seconds = time.perf_counter() - start

    # Everything above used train and validation only. The test split is evaluated here, once.
    confusion = tuple(tuple(int(c) for c in row) for row in model.confusion_matrix(test))
    train_labels = train_file.labels[:cut]
    majority = max(range(CLASS_COUNT), key=train_labels.count)
    baseline = test_file.labels.count(majority) / len(test_file.labels)
    worst = max(
        (
            (confusion[a][p], -a, -p)
            for a in range(CLASS_COUNT)
            for p in range(CLASS_COUNT)
            if a != p
        ),
    )
    metrics = multiclass_metrics(confusion)
    outcome = ExperimentResult(
        (cut, validation_size, len(test_file.labels)),
        validation_counts,
        tuple(candidates),
        selected,
        result.loss_per_epoch,
        seconds,
        model.parameter_count,
        model.accuracy(train),
        model.accuracy(validation),
        metrics,
        confusion,
        majority,
        baseline,
        (-worst[1], -worst[2], worst[0]),
    )
    if check and not (
        metrics.accuracy > baseline
        # A linear model on raw MNIST pixels is well established around 92%; below 90% would mean
        # the optimizer or the preprocessing is broken rather than that the model is weak.
        and metrics.accuracy >= 0.90
        and metrics.macro_f1 >= 0.90
        and result.loss_per_epoch[-1] < result.loss_per_epoch[0]
    ):
        raise RuntimeError(
            f"experiment failed its criteria: test accuracy={metrics.accuracy:.4f}, "
            f"macro F1={metrics.macro_f1:.4f}, baseline={baseline:.4f}"
        )
    return outcome


def run_experiment(directory=None, seed=SEED, **options):
    return run_with(SoftmaxRegression, sparse_mnist, directory, seed, **options)


def print_result(title, seed, result):
    train, validation, test = result.split_sizes
    print(f"{title} softmax regression on MNIST (seed={seed})")
    print(f"splits: train={train}, validation={validation}, test={test}")
    print("validation class counts:", " ".join(map(str, result.validation_counts)))
    print(f"hyperparameter search on the validation split (batch={BATCH_SIZE})")
    for c in result.candidates:
        print(
            f"  learning rate={c.learning_rate:6.4f}, L2={c.l2_regularization:8.4f}: "
            f"train loss={c.final_loss:.4f}, train accuracy={c.train_accuracy:.4f}, "
            f"validation accuracy={c.validation_accuracy:.4f}"
        )
    s = result.selected
    print(f"  selected learning rate={s.learning_rate:.4f}, L2={s.l2_regularization:.4f}")
    history = result.loss_per_epoch
    print(f"final training run ({len(history)} epochs, {result.parameter_count} parameters)")
    for epoch in REPORTED_EPOCHS:
        if epoch <= len(history):
            print(f"  epoch {epoch:2}: training loss={history[epoch - 1]:.4f}")
    print(
        f"  wall-clock training time={result.training_seconds:.2f} s "
        f"({result.training_seconds / len(history):.2f} s per epoch)"
    )
    m = result.test
    print("held-out results")
    print(
        f"  majority-class baseline (always {result.majority_class}): "
        f"test accuracy={result.baseline_accuracy:.4f}"
    )
    print(
        f"  softmax regression: train accuracy={result.train_accuracy:.4f}, "
        f"validation accuracy={result.validation_accuracy:.4f}, test accuracy={m.accuracy:.4f}"
    )
    print(
        f"  macro precision={m.macro_precision:.4f}, macro recall={m.macro_recall:.4f}, "
        f"macro F1={m.macro_f1:.4f}"
    )
    for digit in range(CLASS_COUNT):
        print(
            f"  digit {digit}: precision={m.per_class_precision[digit]:.4f}, "
            f"recall={m.per_class_recall[digit]:.4f}, F1={m.per_class_f1[digit]:.4f}"
        )
    print("test confusion matrix (rows actual, columns predicted)")
    for actual, row in enumerate(result.confusion):
        print(f"  {actual}" + "".join(f"{count:6}" for count in row))
    actual, predicted, count = result.most_frequent_confusion
    print(f"most frequent confusion: {actual} predicted as {predicted}, {count} times")


def main(run=run_experiment, title="Python"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data", default=None, help="MNIST directory (default data/mnist)")
    args = parser.parse_args()
    try:
        result = run(args.data, args.seed)
    except FileNotFoundError as error:
        sys.exit(str(error))
    print_result(title, args.seed, result)


if __name__ == "__main__":
    main()
