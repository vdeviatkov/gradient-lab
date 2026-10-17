"""Fixed splits and validation-only threshold tuning, equivalent to experiment 04 in C++."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from ml_scratch.pure.logistic_regression import (
    BinaryLogisticRegression,
    ClassificationMetrics,
    LogisticTrainingConfig,
    LogisticTrainingResult,
    ThresholdSelectionResult,
    classification_metrics,
    select_threshold_for_f1,
)

SEED = 20260904
TRAINING_DATA = tuple(
    ((x,), 0) for x in (-2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.3, 0.6, 0.9, 1.2)
) + tuple(((x,), 1) for x in (0.4, 1.0, 1.6, 2.2))
VALIDATION_DATA = tuple(((x,), 0) for x in (-1.5, -0.5, 0.2, 0.7, 1.1, 1.5)) + tuple(
    ((x,), 1) for x in (0.5, 0.9, 1.4)
)
TEST_DATA = tuple(((x,), 0) for x in (-2.2, -1.4, -0.7, -0.1, 0.4, 0.8, 1.2, 1.7)) + tuple(
    ((x,), 1) for x in (0.6, 1.5)
)
CANDIDATE_THRESHOLDS = tuple(i / 100 for i in range(15, 76, 5))


@dataclass(frozen=True)
class ExperimentResult:
    training: LogisticTrainingResult
    weights: tuple[float, ...]
    bias: float
    selection: ThresholdSelectionResult
    baseline: ClassificationMetrics
    default: ClassificationMetrics
    selected: ClassificationMetrics


def run_experiment(seed: int = SEED) -> ExperimentResult:
    model = BinaryLogisticRegression(1)
    training = model.fit(
        TRAINING_DATA,
        LogisticTrainingConfig(max_epochs=2_000, batch_size=4, seed=seed, target_loss=0.30),
    )
    selection = select_threshold_for_f1(
        [y for _, y in VALIDATION_DATA],
        [model.predict_probability(x) for x, _ in VALIDATION_DATA],
        CANDIDATE_THRESHOLDS,
    )
    return ExperimentResult(
        training,
        tuple(model.weights),
        model.bias,
        selection,
        classification_metrics([y for _, y in TEST_DATA], [0] * len(TEST_DATA)),
        model.evaluate(TEST_DATA),
        model.evaluate(TEST_DATA, selection.threshold),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    result = run_experiment(args.seed)
    print(f"Pure-Python binary logistic regression experiment (seed={args.seed})")
    print(
        f"epochs={result.training.epochs}, final weighted BCE="
        f"{result.training.loss_per_epoch[-1]:.6f}, weight={result.weights[0]:.6f}, "
        f"bias={result.bias:.6f}"
    )
    print(
        f"validation-selected threshold={result.selection.threshold:.6f}, "
        f"validation F1={result.selection.metrics.f1:.6f}"
    )
    for name, metrics in (
        ("majority baseline", result.baseline),
        ("threshold 0.50", result.default),
        ("selected threshold", result.selected),
    ):
        print(
            f"{name} (test): accuracy={metrics.accuracy:.6f}, precision={metrics.precision:.6f}, "
            f"recall={metrics.recall:.6f}, F1={metrics.f1:.6f}, confusion={metrics.confusion}"
        )
    if not result.training.converged or result.selected.f1 <= max(
        result.default.f1, result.baseline.f1
    ):
        raise RuntimeError("model failed its convergence or held-out F1 criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
