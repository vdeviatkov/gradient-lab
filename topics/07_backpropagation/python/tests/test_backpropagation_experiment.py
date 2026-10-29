import pytest
from ml_scratch.pure.backpropagation import FeedForwardNetwork, Loss
from ml_scratch.pure.backpropagation.experiment import (
    MLP_LAYERS,
    NETWORK_SEED,
    build_spiral,
    majority_baseline,
    run_experiment,
)


def test_spiral_is_reproducible_interleaved_and_balanced():
    data = build_spiral()
    assert data == build_spiral() != build_spiral(1)
    assert [len(s) for s in (data.train, data.validation, data.test)] == [270, 90, 90]
    for split in (data.train, data.validation, data.test):
        counts = [sum(t[k] for _, t in split) for k in range(3)]
        assert counts == [len(split) / 3] * 3
    rows = [{x for x, _ in s} for s in (data.train, data.validation, data.test)]
    assert not rows[0] & rows[1] and not rows[0] & rows[2] and not rows[1] & rows[2]
    assert majority_baseline(data) == pytest.approx(1 / 3)


def test_initialization_reproduces_the_recorded_cpp_network():
    # The C++ README records 9.913e-02 as the MLP's largest initial |gradient|; matching it
    # confirms that the mt19937 draws and the parameter layout are the C++ ones.
    mlp = FeedForwardNetwork(MLP_LAYERS, Loss.SOFTMAX_CROSS_ENTROPY, NETWORK_SEED)
    largest = max(map(abs, mlp.gradient(build_spiral().train)))
    assert f"{largest:.3e}" == "9.913e-02"


def test_short_run_meets_every_experiment_criterion(capsys):
    # The full 4000-epoch run is recorded in the topic README; 100 epochs already satisfy the
    # same criteria and keep the pure-Python suite fast. run_experiment raises otherwise.
    result = run_experiment(epochs=100)
    assert all(c.result.passed for c in result.configurations)
    assert [c.parameters for c in result.configurations] == [6, 27, 22, 17, 51, 103, 81]
    assert result.detection.worst_parameter == 7 and not result.detection.passed
    # Truncation-dominated rows depend only on the parameters, so they match C++ exactly.
    assert f"{result.step_sizes[0].central:.3e}" == "5.329e-04"
    assert f"{result.step_sizes[0].forward:.3e}" == "3.373e-02"
    assert result.mlp.test > result.linear.test > result.baseline
