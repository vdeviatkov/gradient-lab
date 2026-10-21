"""Least-squares optimizers, stability, conditioning, and collinearity with PyTorch."""

import argparse

from ml_scratch.pure.linear_regression.experiment import (
    SEED,
    TEST,
    TRAIN,
    collinear_dataset,
    scaled_dataset,
)

from .model import (
    LinearRegression,
    Optimizer,
    TrainingConfig,
    condition_number,
    hessian_eigenvalue_range,
    standardize,
)


def run_experiment(seed=SEED):
    reference = LinearRegression(1)
    reference.fit_closed_form(TRAIN)
    print(
        f"closed form: weight={reference.weights[0]:.9f}, bias={reference.bias:.9f}, "
        f"test MSE={reference.mean_squared_error(TEST):.3e}"
    )
    results = []
    for optimizer, rate, size in (
        (Optimizer.BATCH, 0.1, 9),
        (Optimizer.STOCHASTIC, 0.03, 1),
        (Optimizer.MINI_BATCH, 0.05, 3),
    ):
        model = LinearRegression(1)
        trained = model.fit(
            TRAIN,
            TrainingConfig(
                optimizer=optimizer,
                learning_rate=rate,
                batch_size=size,
                seed=seed,
                max_epochs=2000,
                target_mse=1e-12,
            ),
        )
        error = model.mean_squared_error(TEST)
        results.append((optimizer.value, trained.epochs, error))
        print(
            f"{optimizer.value}: epochs={trained.epochs}, converged={trained.converged}, "
            f"test MSE={error:.3e}"
        )
        if not trained.converged or error > 1e-10:
            raise RuntimeError("optimizer failed to match closed-form fit")
    _, largest = hessian_eigenvalue_range(TRAIN)
    threshold = 2 / largest
    for multiplier in (0.99, 1.01):
        model = LinearRegression(1)
        trained = model.fit(
            TRAIN, TrainingConfig(learning_rate=multiplier * threshold, max_epochs=1000)
        )
        final = trained.mse_per_epoch[-1]
        print(f"rate={multiplier:.2f} * 2/lambda_max ({threshold:.6f}): MSE={final:.3e}")
        if (multiplier < 1 and final > 1e-10) or (multiplier > 1 and final < 1):
            raise RuntimeError("stability check failed")
    for scale in (1, 10, 100):
        raw = scaled_dataset(scale)
        for name, data in (("raw", raw), ("standardized", standardize(raw))):
            _, largest = hessian_eigenvalue_range(data)
            trained = LinearRegression(2).fit(
                data,
                TrainingConfig(learning_rate=1.8 / largest, max_epochs=200_000, target_mse=1e-10),
            )
            print(
                f"scale={scale}, {name}: condition={condition_number(data):.2f}, "
                f"epochs={trained.epochs}, converged={trained.converged}"
            )
            if not trained.converged:
                raise RuntimeError("conditioning run failed to converge")
    for separation in (1e-2, 1e-4, 1e-6, 1e-8):
        data = collinear_dataset(separation, seed=seed)
        model = LinearRegression(2)
        try:
            model.fit_closed_form(data)
        except ValueError:
            print(f"separation={separation:.0e}: singular to working precision")
        else:
            print(
                f"separation={separation:.0e}: weights={model.weights}, "
                f"MSE={model.mean_squared_error(data):.3e}"
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(f"PyTorch linear regression experiment (seed={args.seed})")
    run_experiment(args.seed)


if __name__ == "__main__":
    main()
