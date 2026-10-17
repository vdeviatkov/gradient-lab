import math

import pytest
from ml_scratch.pure.linear_regression.experiment import (
    collinear_dataset,
    fit_methods,
    main,
    match_cpp_convention,
    measure_collinearity,
    measure_conditioning,
    measure_stability,
    run_experiment,
    scaled_dataset,
)


def test_every_method_recovers_the_generating_line() -> None:
    summaries = fit_methods()

    assert tuple(summary.name for summary in summaries) == (
        "closed form",
        "batch GD",
        "SGD",
        "mini-batch GD",
    )
    for summary in summaries:
        assert summary.weight == pytest.approx(2.0, abs=1e-5)
        assert summary.bias == pytest.approx(1.5, abs=1e-5)
        assert summary.train_mse < 1e-10
        assert summary.test_mse < 1e-10

    # Only the closed form answers without epochs.
    assert summaries[0].epochs is None
    assert all(summary.epochs is not None for summary in summaries[1:])


def test_halving_the_rate_reproduces_the_cpp_epoch_counts() -> None:
    """The C++ milestone omits the factor of two in its gradient; halving ours should match it.

    If this ever fails, the two implementations differ by more than a scale factor and the
    cross-implementation comparison in the experiment README needs revisiting.
    """
    for name, ours, theirs in match_cpp_convention():
        assert ours == theirs, name


def test_measured_stability_threshold_matches_the_prediction() -> None:
    summary = measure_stability(max_epochs=500, steps=40)

    assert summary.predicted_threshold == pytest.approx(2.0 / summary.largest_eigenvalue)
    assert summary.relative_error < 1e-3
    assert summary.measured_threshold == pytest.approx(summary.predicted_threshold, rel=1e-3)


def test_scaled_dataset_keeps_the_relationship_exact() -> None:
    for scale in (1.0, 10.0, 100.0):
        dataset = scaled_dataset(scale)
        for (first, second), target in dataset:
            assert target == pytest.approx(1.5 + 2.0 * first - 0.5 * second)


def test_worse_conditioning_costs_more_epochs_and_standardizing_pays_it_back() -> None:
    easy = measure_conditioning(1.0)
    hard = measure_conditioning(100.0)

    assert easy.converged and hard.converged
    assert hard.condition > 100.0 * easy.condition
    assert hard.epochs > 100 * easy.epochs
    # Standardizing restores the easy problem's conditioning and most of its speed.
    assert hard.standardized_condition < 2.0
    assert hard.standardized_epochs < hard.epochs // 1000
    assert hard.standardized_converged


def test_collinear_dataset_has_the_requested_separation() -> None:
    dataset = collinear_dataset(1e-3, seed=0)

    for first, second in (features for features, _ in dataset):
        assert abs(second - first) <= 1e-3


def test_near_collinearity_destroys_the_coefficients_but_not_the_fit() -> None:
    loose = measure_collinearity(1e-2)
    tight = measure_collinearity(1e-6)

    assert not loose.singular and not tight.singular
    # The individual coefficients move enormously...
    assert abs(tight.weights[1]) > 100.0 * abs(loose.weights[1])
    # ...while their sum, which is what the data determines, does not.
    assert tight.weight_sum == pytest.approx(loose.weight_sum, abs=1e-3)
    # ...and neither does the quality of the fit.
    assert tight.train_mse == pytest.approx(loose.train_mse, rel=1e-6)


def test_an_exactly_collinear_design_is_refused() -> None:
    summary = measure_collinearity(0.0)

    assert summary.singular
    assert summary.weights == ()
    assert math.isnan(summary.train_mse)


def test_run_experiment_returns_every_part_and_is_deterministic() -> None:
    first = run_experiment(scales=(1.0, 100.0), separations=(1e-2, 1e-6))
    second = run_experiment(scales=(1.0, 100.0), separations=(1e-2, 1e-6))

    methods, stability, conditioning, collinearity = first
    assert len(methods) == 4
    assert len(conditioning) == 2
    assert len(collinearity) == 2
    assert stability.relative_error < 1e-3
    assert first == second


def test_main_prints_every_part_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    for part in ("part one", "part two", "part three", "part four"):
        assert part in output
    assert "closed form" in output
    assert "singular to working precision" in output
