import numpy as np
import pytest
from ml_scratch.numpy.linear_regression import (
    LinearRegression,
    Optimizer,
    TrainingConfig,
    condition_number,
    hessian,
    hessian_eigenvalue_range,
    solve_linear_system,
    standardize,
)
from ml_scratch.numpy.linear_regression.experiment import run_experiment
from ml_scratch.pure.linear_regression import LinearRegression as ScalarRegression
from ml_scratch.pure.linear_regression.experiment import TEST, TRAIN

DATA = (((-1, 2), 3), ((0, -1), 0), ((1, 1), 5), ((2, 0), 6))


def test_closed_form_elimination_pivoting_and_singular_rejection():
    np.testing.assert_allclose(solve_linear_system([[0, 2], [1, 3]], [4, 7]), [1, 2])
    model, scalar = LinearRegression(2), ScalarRegression(2)
    model.fit_closed_form(DATA)
    scalar.fit_closed_form(DATA)
    np.testing.assert_allclose(model.weights, scalar.weights, atol=1e-12)
    assert model.bias == pytest.approx(scalar.bias)
    assert model.r_squared(DATA) == pytest.approx(scalar.r_squared(DATA))
    with pytest.raises(ValueError):
        model.fit_closed_form((((0, 0), 0), ((1, 1), 1)))
    with pytest.raises(ValueError):
        solve_linear_system([[1, 2]], [0])


def test_gradient_against_central_differences_and_scalar():
    model = LinearRegression(2)
    model.weights[:], model.bias = [0.4, -0.2], 0.7
    scalar = ScalarRegression(2)
    scalar.weights, scalar.bias = list(model.weights), model.bias
    weights, bias = model.gradient(DATA)
    ref_w, ref_b = scalar.gradient(DATA)
    np.testing.assert_allclose(weights, ref_w, atol=1e-14)
    assert bias == pytest.approx(ref_b)
    eps = 1e-6
    for index in range(3):
        original = model.weights[index] if index < 2 else model.bias
        losses = []
        for sign in (1, -1):
            if index < 2:
                model.weights[index] = original + sign * eps
            else:
                model.bias = original + sign * eps
            losses.append(model.mean_squared_error(DATA))
        if index < 2:
            model.weights[index] = original
        else:
            model.bias = original
        assert (losses[0] - losses[1]) / (2 * eps) == pytest.approx(
            weights[index] if index < 2 else bias, abs=1e-8
        )


@pytest.mark.parametrize("optimizer", list(Optimizer))
def test_optimizer_training_matches_scalar(optimizer):
    config = TrainingConfig(
        optimizer=optimizer, learning_rate=0.02, max_epochs=40, batch_size=3, seed=29
    )
    array, scalar = LinearRegression(2), ScalarRegression(2)
    result, reference = array.fit(DATA, config), scalar.fit(DATA, config)
    np.testing.assert_allclose(result.mse_per_epoch, reference.mse_per_epoch, atol=1e-13)
    np.testing.assert_allclose(array.weights, scalar.weights, atol=1e-13)
    again = LinearRegression(2)
    assert again.fit(DATA, config) == result


def test_hessian_standardization_and_stability():
    smallest, largest = hessian_eigenvalue_range(TRAIN)
    np.testing.assert_allclose([smallest, largest], np.linalg.eigvalsh(hessian(TRAIN))[[0, -1]])
    transformed = standardize(DATA)
    x = np.array([row for row, _ in transformed])
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(x.std(axis=0), 1)
    assert condition_number(TRAIN) > 1
    stable = LinearRegression(1).fit(
        TRAIN, TrainingConfig(learning_rate=1.8 / largest, max_epochs=1000, target_mse=1e-12)
    )
    assert stable.converged
    diverged = LinearRegression(1).fit(TRAIN, TrainingConfig(learning_rate=100, max_epochs=1000))
    assert diverged.diverged and not diverged.converged


@pytest.mark.parametrize("data", [(), (((1,), 0),), (((np.nan, 2), 0),), (((1, 2), np.inf),)])
def test_invalid_data(data):
    with pytest.raises(ValueError):
        LinearRegression(2).gradient(data)


def test_invalid_config_and_constant_targets():
    for config in (
        TrainingConfig(learning_rate=np.inf),
        TrainingConfig(max_epochs=0),
        TrainingConfig(batch_size=0),
        TrainingConfig(target_mse=-1),
    ):
        with pytest.raises(ValueError):
            LinearRegression(1).fit(TRAIN, config)
    with pytest.raises(ValueError):
        LinearRegression(1).r_squared((((1,), 1), ((2,), 1)))


def test_linear_experiment(capsys):
    reports = run_experiment()
    assert len(reports) == 3 and max(r[2] for r in reports) < 1e-10
    assert "standardized" in capsys.readouterr().out
    model = LinearRegression(1)
    model.fit_closed_form(TRAIN)
    assert model.mean_squared_error(TEST) < 1e-25
