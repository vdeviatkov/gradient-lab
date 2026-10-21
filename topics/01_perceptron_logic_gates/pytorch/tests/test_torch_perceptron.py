import random

import pytest
import torch
from ml_scratch.pure.perceptron import Perceptron as ScalarPerceptron
from ml_scratch.pytorch.perceptron import Perceptron
from ml_scratch.pytorch.perceptron.experiment import run_experiment

from ml_scratch.pure.datasets import logic_gate_dataset


@pytest.mark.parametrize("gate", ["and", "or", "xor"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_matches_scalar_online_training(gate, shuffle):
    data = logic_gate_dataset(gate)
    model, reference = Perceptron(2, seed=7), ScalarPerceptron(2, seed=7)
    assert model.fit(data, shuffle=shuffle) == reference.fit(data, shuffle=shuffle)
    torch.testing.assert_close(model.weights, torch.tensor(reference.weights, dtype=torch.float64))
    assert model.bias == pytest.approx(reference.bias)
    assert model.predict_many(x for x, _ in data).tolist() == list(
        reference.predict_many(x for x, _ in data)
    )
    assert model.weights.dtype == torch.float64 and model.weights.device.type == "cpu"


def test_hand_update_zero_boundary_and_no_global_rng_changes():
    python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
    model = Perceptron(2, learning_rate=0.2)
    model.weights.zero_()
    model.bias = 0.0
    assert model.predict([0, 0]) == 1
    model.fit((((2, -1), 0),), max_epochs=1, shuffle=False)
    torch.testing.assert_close(model.weights, torch.tensor([-0.4, 0.2], dtype=torch.float64))
    assert model.bias == pytest.approx(-0.2)
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("data", [(), (((0,), 0),), (((0, float("nan")), 0),), (((0, 0), 2),)])
def test_invalid_inputs(data):
    with pytest.raises(ValueError):
        Perceptron(2).fit(data)


def test_invalid_controls_and_experiment(capsys):
    for kwargs in ({"feature_count": 0}, {"feature_count": 2, "learning_rate": float("inf")}):
        with pytest.raises(ValueError):
            Perceptron(**kwargs)
    with pytest.raises(ValueError):
        Perceptron(2).fit(logic_gate_dataset("and"), max_epochs=0)
    results = run_experiment()
    assert results[0][1].converged and not results[-1][1].converged
    assert "XOR" in capsys.readouterr().out
