import numpy as np
import pytest
from ml_scratch.numpy.perceptron import Perceptron
from ml_scratch.numpy.perceptron.experiment import run_experiment
from ml_scratch.pure.perceptron import Perceptron as ScalarPerceptron

from ml_scratch.pure.datasets import logic_gate_dataset


@pytest.mark.parametrize("gate", ["and", "or", "xor"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_matches_scalar_training(gate, shuffle):
    data = logic_gate_dataset(gate)
    array, scalar = Perceptron(2, seed=7), ScalarPerceptron(2, seed=7)
    result = array.fit(data, shuffle=shuffle)
    reference = scalar.fit(data, shuffle=shuffle)
    assert result == reference
    np.testing.assert_allclose(array.weights, scalar.weights, atol=1e-14)
    assert array.bias == pytest.approx(scalar.bias)
    np.testing.assert_array_equal(
        array.predict_many(x for x, _ in data), scalar.predict_many(x for x, _ in data)
    )


def test_hand_computed_online_update_and_boundary():
    model = Perceptron(2, learning_rate=0.2)
    model.weights[:], model.bias = 0, 0
    assert model.predict([0, 0]) == 1
    model.fit((((2, -1), 0),), max_epochs=1, shuffle=False)
    np.testing.assert_allclose(model.weights, [-0.4, 0.2])
    assert model.bias == pytest.approx(-0.2)


@pytest.mark.parametrize("data", [(), (((0,), 0),), (((0, np.nan), 0),), (((0, 0), 2),)])
def test_invalid_data(data):
    with pytest.raises(ValueError):
        Perceptron(2).fit(data)


def test_invalid_config_and_experiment(capsys):
    for kwargs in ({"feature_count": 0}, {"feature_count": 2, "learning_rate": np.inf}):
        with pytest.raises(ValueError):
            Perceptron(**kwargs)
    with pytest.raises(ValueError):
        Perceptron(2).fit(logic_gate_dataset("and"), max_epochs=0)
    results = run_experiment()
    assert results[0][1].converged and not results[-1][1].converged
    assert "XOR" in capsys.readouterr().out
