from typing import Any

from ml_scratch.pure.perceptron_experiment import main, run_experiment


def test_experiment_learns_and_or_but_not_xor() -> None:
    and_summary, or_summary, xor_summary = run_experiment(seed=7, max_epochs=100)

    assert and_summary.converged
    assert and_summary.predictions == and_summary.targets
    assert or_summary.converged
    assert or_summary.predictions == or_summary.targets
    assert not xor_summary.converged
    assert xor_summary.predictions != xor_summary.targets


def test_cli_describes_xor_limitation(capsys: Any) -> None:
    assert main(["--seed", "7", "--max-epochs", "100"]) == 0
    output = capsys.readouterr().out

    assert "AND: learned" in output
    assert "OR: learned" in output
    assert "XOR: not linearly separable" in output
    assert "XOR: learned" not in output
