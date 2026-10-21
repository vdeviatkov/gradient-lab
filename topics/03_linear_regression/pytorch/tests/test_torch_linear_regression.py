import pytest
import torch
from ml_scratch.pure.linear_regression import LinearRegression as ScalarRegression
from ml_scratch.pure.linear_regression.experiment import TEST, TRAIN
from ml_scratch.pytorch.linear_regression import (
    LinearRegression,
    Optimizer,
    TrainingConfig,
    condition_number,
    hessian,
    hessian_eigenvalue_range,
    solve_linear_system,
    standardize,
)
from ml_scratch.pytorch.linear_regression.experiment import run_experiment

DATA = (((-1, 2), 3), ((0.2, -1), 0), ((1, 1), 5), ((2, 0), 6))


def test_elimination_pivoting_normal_equations_and_singularity():
    torch.testing.assert_close(
        solve_linear_system([[0, 2], [1, 3]], [4, 7]), torch.tensor([1, 2], dtype=torch.float64)
    )
    model, scalar = LinearRegression(2), ScalarRegression(2)
    model.fit_closed_form(DATA)
    scalar.fit_closed_form(DATA)
    torch.testing.assert_close(model.weights, torch.tensor(scalar.weights, dtype=torch.float64))
    assert model.r_squared(DATA) == pytest.approx(scalar.r_squared(DATA))
    with pytest.raises(ValueError):
        model.fit_closed_form((((0, 0), 0), ((1, 1), 1)))
    with pytest.raises(ValueError):
        solve_linear_system([[1, 2]], [0])


def test_manual_gradient_matches_autograd_scalar_and_finite_differences():
    model, scalar = LinearRegression(2), ScalarRegression(2)
    model.weights[:] = torch.tensor([0.4, -0.2], dtype=torch.float64)
    model.bias = 0.7
    scalar.weights, scalar.bias = model.weights.tolist(), model.bias
    w, b = model.gradient(DATA)
    auto_w, auto_b = model.autograd_gradient(DATA)
    ref_w, ref_b = scalar.gradient(DATA)
    torch.testing.assert_close(w, auto_w)
    torch.testing.assert_close(w, torch.tensor(ref_w, dtype=torch.float64))
    assert b == pytest.approx(auto_b) == pytest.approx(ref_b)
    eps = 1e-6
    for index in range(3):
        original = float(model.weights[index]) if index < 2 else model.bias
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
            float(w[index]) if index < 2 else b, abs=1e-8
        )
    assert model.weights.grad is None


@pytest.mark.parametrize("optimizer", list(Optimizer))
def test_training_matches_scalar_and_repeats(optimizer):
    config = TrainingConfig(
        optimizer=optimizer, learning_rate=0.02, max_epochs=40, batch_size=3, seed=29
    )
    model, scalar = LinearRegression(2), ScalarRegression(2)
    result, reference = model.fit(DATA, config), scalar.fit(DATA, config)
    torch.testing.assert_close(
        torch.tensor(result.mse_per_epoch), torch.tensor(reference.mse_per_epoch)
    )
    torch.testing.assert_close(model.weights, torch.tensor(scalar.weights, dtype=torch.float64))
    assert LinearRegression(2).fit(DATA, config) == result


def test_hessian_scaling_and_divergence():
    smallest, largest = hessian_eigenvalue_range(TRAIN)
    torch.testing.assert_close(
        torch.tensor([smallest, largest], dtype=torch.float64),
        torch.linalg.eigvalsh(hessian(TRAIN))[[0, -1]],
    )
    x = torch.tensor([row for row, _ in standardize(DATA)], dtype=torch.float64)
    torch.testing.assert_close(
        x.mean(dim=0), torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0
    )
    torch.testing.assert_close(x.std(dim=0, correction=0), torch.ones(2, dtype=torch.float64))
    assert condition_number(TRAIN) > 1
    result = LinearRegression(1).fit(TRAIN, TrainingConfig(learning_rate=100, max_epochs=1000))
    assert result.diverged and not result.converged


@pytest.mark.parametrize(
    "data", [(), (((1,), 0),), (((1, float("nan")), 0),), (((1, 2), float("inf")),)]
)
def test_invalid_inputs(data):
    with pytest.raises(ValueError):
        LinearRegression(2).fit(data)


def test_invalid_controls_and_experiment(capsys):
    for config in (
        TrainingConfig(learning_rate=float("inf")),
        TrainingConfig(max_epochs=0),
        TrainingConfig(batch_size=0),
        TrainingConfig(target_mse=-1),
    ):
        with pytest.raises(ValueError):
            LinearRegression(1).fit(TRAIN, config)
    with pytest.raises(ValueError):
        LinearRegression(1).r_squared((((1,), 1), ((2,), 1)))
    reports = run_experiment()
    assert len(reports) == 3 and max(r[2] for r in reports) < 1e-10
    assert "standardized" in capsys.readouterr().out
    model = LinearRegression(1)
    model.fit_closed_form(TRAIN)
    assert model.mean_squared_error(TEST) < 1e-25
