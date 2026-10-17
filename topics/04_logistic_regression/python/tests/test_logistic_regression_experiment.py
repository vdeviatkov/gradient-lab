from ml_scratch.pure.logistic_regression.experiment import SEED, main, run_experiment


def test_experiment_converges_and_improves_held_out_f1():
    result = run_experiment()
    assert result == run_experiment(SEED)
    assert result.training.converged
    assert result.training.loss_per_epoch[-1] <= 0.30
    assert result.baseline.accuracy == 0.8
    assert result.baseline.f1 == 0
    assert result.selected.f1 > result.default.f1
    assert result.selected.f1 > result.baseline.f1


def test_cli_reports_seed_and_metrics(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert f"seed={SEED}" in output
    assert "validation-selected threshold=" in output
    assert "selected threshold (test):" in output
