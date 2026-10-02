import pytest

from ml_scratch.pure.xor_mlp_experiment import (
    gradient_check,
    main,
    run_experiment,
    sweep_width,
    train_once,
)

_FAST = {"learning_rate": 0.5, "max_epochs": 2_000, "target_loss": 0.02}


def test_headline_run_learns_xor() -> None:
    model, summary = train_once(4, seed=7, **_FAST)

    assert summary.converged
    assert summary.accuracy == 1.0
    assert summary.final_loss <= 0.02
    assert model.predict_many(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))) == (0, 1, 1, 0)


def test_one_hidden_unit_never_solves_xor() -> None:
    summary = sweep_width(1, seeds=8, **_FAST)

    assert summary.solved == 0
    assert summary.median_epochs is None
    # Every failure is a single linear boundary doing the best one can on XOR, or worse.
    assert max(summary.failure_accuracies) <= 0.75


def test_two_hidden_units_sometimes_solve_xor() -> None:
    """The theoretical minimum width works, but only from some initializations."""
    summary = sweep_width(2, seeds=16, **_FAST)

    assert 0 < summary.solved < summary.attempts
    assert summary.median_epochs is not None
    assert summary.failure_accuracies == (0.5,)


def test_four_hidden_units_always_solved_xor() -> None:
    summary = sweep_width(4, seeds=16, **_FAST)

    assert summary.solved == summary.attempts
    assert summary.solved_fraction == 1.0
    assert summary.failure_accuracies == ()
    assert summary.failure_losses == ()


def test_wider_is_not_slower_in_epochs() -> None:
    narrow = sweep_width(2, seeds=16, **_FAST)
    wide = sweep_width(8, seeds=16, **_FAST)

    assert narrow.median_epochs is not None
    assert wide.median_epochs is not None
    assert wide.median_epochs <= narrow.median_epochs


def test_gradient_check_is_tight() -> None:
    assert gradient_check(4, seed=7) < 1e-9


def test_run_experiment_returns_every_part() -> None:
    headline, sweeps, largest_difference = run_experiment(seeds=8, widths=(1, 2, 4), **_FAST)

    assert headline.converged
    assert tuple(sweep.hidden_units for sweep in sweeps) == (1, 2, 4)
    assert largest_difference < 1e-9


def test_experiment_is_deterministic() -> None:
    first = run_experiment(seeds=8, widths=(1, 4), **_FAST)
    second = run_experiment(seeds=8, widths=(1, 4), **_FAST)

    assert first == second


def test_main_prints_every_part_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--seeds", "8", "--max-epochs", "2000"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "part one" in output
    assert "part two" in output
    assert "learned in" in output
    assert "central differences agree" in output
