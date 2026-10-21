"""Train on fixed data and choose the decision threshold using validation F1 only."""

import argparse

from ml_scratch.pure.logistic_regression.experiment import (
    CANDIDATE_THRESHOLDS,
    SEED,
    TEST_DATA,
    TRAINING_DATA,
    VALIDATION_DATA,
    ExperimentResult,
)

from .model import (
    BinaryLogisticRegression,
    LogisticTrainingConfig,
    classification_metrics,
    select_threshold_for_f1,
)


def run_experiment(seed=SEED):
    model = BinaryLogisticRegression(1)
    trained = model.fit(
        TRAINING_DATA,
        LogisticTrainingConfig(max_epochs=2000, batch_size=4, seed=seed, target_loss=0.30),
    )
    selection = select_threshold_for_f1(
        [y for _, y in VALIDATION_DATA],
        model.predict_probabilities([x for x, _ in VALIDATION_DATA]),
        CANDIDATE_THRESHOLDS,
    )
    result = ExperimentResult(
        trained,
        tuple(model.weights.tolist()),
        model.bias,
        selection,
        classification_metrics([y for _, y in TEST_DATA], [0] * len(TEST_DATA)),
        model.evaluate(TEST_DATA),
        model.evaluate(TEST_DATA, selection.threshold),
    )
    if not trained.converged or result.selected.f1 <= max(result.default.f1, result.baseline.f1):
        raise RuntimeError("logistic experiment failed its held-out F1 or convergence criteria")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(f"PyTorch logistic regression experiment (seed={args.seed})")
    result = run_experiment(args.seed)
    print(
        f"epochs={result.training.epochs}, BCE={result.training.loss_per_epoch[-1]:.6f}, "
        f"weight={result.weights[0]:.6f}, bias={result.bias:.6f}"
    )
    print(
        f"validation-selected threshold={result.selection.threshold:.2f}, "
        f"validation F1={result.selection.metrics.f1:.6f}"
    )
    for name in ("baseline", "default", "selected"):
        print(f"{name} test: {getattr(result, name)}")


if __name__ == "__main__":
    main()
