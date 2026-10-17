import math

import pytest
from ml_scratch.pure.linear_regression import (
    LinearRegression,
    Optimizer,
    TrainingConfig,
    condition_number,
    hessian,
    hessian_eigenvalue_range,
    largest_hessian_eigenvalue,
    solve_linear_system,
    standardize,
)

EXACT_LINE = tuple(((x,), 1.5 + 2.0 * x) for x in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0))


# -- the linear solver ---------------------------------------------------------------------


def test_solver_matches_a_hand_computed_solution() -> None:
    # 2a + b = 5, a + 3b = 10  ->  a = 1, b = 3
    assert solve_linear_system([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0]) == pytest.approx([1.0, 3.0])


def test_solver_needs_pivoting_to_handle_a_zero_leading_entry() -> None:
    """Without a row swap the elimination would divide by the leading zero."""
    solution = solve_linear_system([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0])

    assert solution == pytest.approx([3.0, 2.0])


def test_solver_refuses_a_singular_system() -> None:
    with pytest.raises(ValueError, match="singular"):
        solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [3.0, 6.0])


def test_solver_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="square"):
        solve_linear_system([[1.0, 2.0]], [1.0, 2.0])


# -- the model -----------------------------------------------------------------------------


def test_prediction_is_the_affine_combination() -> None:
    model = LinearRegression(2)
    model.weights = [2.0, -3.0]
    model.bias = 0.5

    assert model.predict((1.0, 2.0)) == pytest.approx(2.0 - 6.0 + 0.5)
    assert model.predict_many([(0.0, 0.0), (1.0, 1.0)]) == pytest.approx([0.5, -0.5])


def test_metrics_match_their_definitions() -> None:
    model = LinearRegression(1)
    model.weights = [1.0]
    model.bias = 0.0
    dataset = (((0.0,), 1.0), ((2.0,), 1.0))

    # Predictions 0 and 2 against targets 1 and 1: squared errors 1 and 1.
    assert model.mean_squared_error(dataset) == pytest.approx(1.0)
    # The targets are identical, so their variance is zero and r_squared is undefined.
    with pytest.raises(ValueError, match="undefined"):
        model.r_squared(dataset)

    varied = (((0.0,), 0.0), ((2.0,), 2.0))
    assert model.r_squared(varied) == pytest.approx(1.0)


def test_mean_squared_error_returns_infinity_instead_of_raising_on_overflow() -> None:
    """CPython's ``**`` raises on float overflow where ``*`` returns infinity; a diverged run
    must report infinity rather than blow up its caller."""
    model = LinearRegression(1)
    model.weights = [1e200]
    model.bias = 0.0

    assert model.mean_squared_error((((1e200,), 0.0),)) == math.inf


def test_gradient_matches_central_differences() -> None:
    model = LinearRegression(2)
    model.weights = [0.3, -0.7]
    model.bias = 0.2
    dataset = (((1.0, 2.0), 3.0), ((-1.0, 0.5), 0.25), ((2.0, -1.0), 1.0))

    weight_gradient, bias_gradient = model.gradient(dataset)

    epsilon = 1e-6
    for index in range(2):
        original = model.weights[index]
        model.weights[index] = original + epsilon
        above = model.mean_squared_error(dataset)
        model.weights[index] = original - epsilon
        below = model.mean_squared_error(dataset)
        model.weights[index] = original
        assert weight_gradient[index] == pytest.approx((above - below) / (2 * epsilon), abs=1e-7)

    model.bias += epsilon
    above = model.mean_squared_error(dataset)
    model.bias -= 2 * epsilon
    below = model.mean_squared_error(dataset)
    model.bias += epsilon
    assert bias_gradient == pytest.approx((above - below) / (2 * epsilon), abs=1e-7)


def test_closed_form_recovers_an_exact_line() -> None:
    model = LinearRegression(1)

    model.fit_closed_form(EXACT_LINE)

    assert model.weights[0] == pytest.approx(2.0, abs=1e-12)
    assert model.bias == pytest.approx(1.5, abs=1e-12)
    assert model.mean_squared_error(EXACT_LINE) == pytest.approx(0.0, abs=1e-24)


def test_closed_form_is_the_least_squares_minimum() -> None:
    """Perturbing the fitted coefficients in any direction must increase the error."""
    noisy = (((0.0,), 0.1), ((1.0,), 1.9), ((2.0,), 4.2), ((3.0,), 5.8))
    model = LinearRegression(1)
    model.fit_closed_form(noisy)
    best = model.mean_squared_error(noisy)

    for delta in (1e-3, -1e-3):
        model.weights[0] += delta
        assert model.mean_squared_error(noisy) > best
        model.weights[0] -= delta
        model.bias += delta
        assert model.mean_squared_error(noisy) > best
        model.bias -= delta

    # The gradient vanishes there, which is the same statement.
    weight_gradient, bias_gradient = model.gradient(noisy)
    assert weight_gradient[0] == pytest.approx(0.0, abs=1e-12)
    assert bias_gradient == pytest.approx(0.0, abs=1e-12)


def test_closed_form_refuses_an_exactly_collinear_design() -> None:
    collinear = (((1.0, 2.0), 1.0), ((2.0, 4.0), 2.0), ((3.0, 6.0), 3.0))
    model = LinearRegression(2)

    with pytest.raises(ValueError, match="singular"):
        model.fit_closed_form(collinear)


@pytest.mark.parametrize(
    ("optimizer", "learning_rate", "batch_size"),
    [
        (Optimizer.BATCH, 0.10, 9),
        (Optimizer.STOCHASTIC, 0.03, 1),
        (Optimizer.MINI_BATCH, 0.05, 3),
    ],
)
def test_every_gradient_method_reaches_the_closed_form_answer(
    optimizer: Optimizer, learning_rate: float, batch_size: int
) -> None:
    reference = LinearRegression(1)
    reference.fit_closed_form(EXACT_LINE)

    model = LinearRegression(1)
    result = model.fit(
        EXACT_LINE,
        TrainingConfig(
            optimizer=optimizer,
            learning_rate=learning_rate,
            max_epochs=2_000,
            batch_size=batch_size,
            seed=20260903,
            target_mse=1e-12,
        ),
    )

    assert result.converged
    assert not result.diverged
    assert model.weights[0] == pytest.approx(reference.weights[0], abs=1e-5)
    assert model.bias == pytest.approx(reference.bias, abs=1e-5)


def test_batch_gradient_descent_makes_one_update_per_epoch() -> None:
    """Batch uses every example at once, so one epoch is one step down the exact gradient."""
    model = LinearRegression(1)
    expected = LinearRegression(1)
    weight_gradient, bias_gradient = expected.gradient(EXACT_LINE)
    expected.weights[0] -= 0.1 * weight_gradient[0]
    expected.bias -= 0.1 * bias_gradient

    model.fit(EXACT_LINE, TrainingConfig(learning_rate=0.1, max_epochs=1, target_mse=0.0))

    assert model.weights[0] == pytest.approx(expected.weights[0])
    assert model.bias == pytest.approx(expected.bias)


def test_divergence_is_reported_rather_than_returned_as_a_number() -> None:
    model = LinearRegression(1)

    result = model.fit(EXACT_LINE, TrainingConfig(learning_rate=5.0, max_epochs=5_000))

    assert result.diverged
    assert not result.converged
    assert not math.isfinite(result.mse_per_epoch[-1])


def test_same_seed_reproduces_a_stochastic_run() -> None:
    config = TrainingConfig(
        optimizer=Optimizer.STOCHASTIC, learning_rate=0.03, max_epochs=50, seed=5
    )
    first = LinearRegression(1)
    second = LinearRegression(1)
    other = LinearRegression(1)

    first_result = first.fit(EXACT_LINE, config)
    second_result = second.fit(EXACT_LINE, config)
    other_result = other.fit(
        EXACT_LINE,
        TrainingConfig(optimizer=Optimizer.STOCHASTIC, learning_rate=0.03, max_epochs=50, seed=6),
    )

    assert first_result == second_result
    assert first.weights == second.weights
    assert other_result.mse_per_epoch != first_result.mse_per_epoch


def test_invalid_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="feature_count"):
        LinearRegression(0)

    model = LinearRegression(1)
    with pytest.raises(ValueError, match="expected 1 features"):
        model.predict((1.0, 2.0))
    with pytest.raises(ValueError, match="must not be empty"):
        model.mean_squared_error(())
    with pytest.raises(ValueError, match="learning_rate"):
        model.fit(EXACT_LINE, TrainingConfig(learning_rate=0.0))
    with pytest.raises(ValueError, match="max_epochs"):
        model.fit(EXACT_LINE, TrainingConfig(max_epochs=0))
    with pytest.raises(ValueError, match="batch_size"):
        model.fit(EXACT_LINE, TrainingConfig(batch_size=0))
    with pytest.raises(ValueError, match="target_mse"):
        model.fit(EXACT_LINE, TrainingConfig(target_mse=-1.0))


# -- the Hessian ---------------------------------------------------------------------------


def test_hessian_matches_its_definition() -> None:
    dataset = (((1.0,), 0.0), ((3.0,), 0.0))
    # X~ = [[1, 1], [1, 3]]; (2/n) X~^T X~ = [[2, 4], [4, 10]].
    matrix = hessian(dataset)
    assert matrix[0] == pytest.approx([2.0, 4.0])
    assert matrix[1] == pytest.approx([4.0, 10.0])


def test_power_iteration_matches_the_exact_two_by_two_eigenvalues() -> None:
    dataset = (((1.0,), 0.0), ((3.0,), 0.0), ((-2.0,), 0.0))
    matrix = hessian(dataset)
    first, second = matrix[0][0], matrix[1][1]
    off = matrix[0][1]
    gap = math.sqrt((first - second) ** 2 + 4.0 * off * off)
    exact_largest = 0.5 * ((first + second) + gap)
    exact_smallest = 0.5 * ((first + second) - gap)

    smallest, largest = hessian_eigenvalue_range(dataset)

    assert largest == pytest.approx(exact_largest, rel=1e-12)
    assert smallest == pytest.approx(exact_smallest, rel=1e-9)
    assert largest_hessian_eigenvalue(dataset) == pytest.approx(exact_largest, rel=1e-12)
    assert condition_number(dataset) == pytest.approx(exact_largest / exact_smallest, rel=1e-9)


def test_the_stability_threshold_predicts_where_descent_stops_working() -> None:
    """Below ``2 / lambda_max`` the error falls; above it, the error grows."""
    threshold = 2.0 / largest_hessian_eigenvalue(EXACT_LINE)

    below = LinearRegression(1).fit(
        EXACT_LINE, TrainingConfig(learning_rate=0.95 * threshold, max_epochs=500)
    )
    above = LinearRegression(1).fit(
        EXACT_LINE, TrainingConfig(learning_rate=1.05 * threshold, max_epochs=500)
    )

    assert below.mse_per_epoch[-1] < below.mse_per_epoch[0]
    assert above.mse_per_epoch[-1] > above.mse_per_epoch[0]


def test_standardizing_removes_the_conditioning_penalty() -> None:
    skewed = tuple(((x, 100.0 * x * x), 1.0 + x) for x in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0))

    assert condition_number(skewed) > 100.0
    assert condition_number(standardize(skewed)) < 10.0


def test_standardizing_leaves_the_targets_and_shape_alone() -> None:
    dataset = (((1.0, 10.0), 5.0), ((3.0, 30.0), 6.0), ((5.0, 50.0), 7.0))

    standardized = standardize(dataset)

    assert [target for _, target in standardized] == [5.0, 6.0, 7.0]
    assert len(standardized) == len(dataset)
    for index in range(2):
        column = [features[index] for features, _ in standardized]
        assert sum(column) / len(column) == pytest.approx(0.0, abs=1e-12)
        assert sum(value * value for value in column) / len(column) == pytest.approx(1.0)


def test_standardizing_tolerates_a_constant_feature() -> None:
    """A feature with no spread cannot be scaled to unit variance; it must not divide by zero."""
    dataset = (((1.0, 7.0), 1.0), ((2.0, 7.0), 2.0))

    standardized = standardize(dataset)

    assert all(math.isfinite(value) for features, _ in standardized for value in features)
