import numpy as np
import pytest
from ml_scratch.numpy.mlp import MlpParameters, XorMlp
from ml_scratch.numpy.mlp.experiment import gradient_check, run_experiment
from ml_scratch.pure.mlp import XorMlp as ScalarMlp

from ml_scratch.pure.datasets import logic_gate_dataset

DATA = logic_gate_dataset("xor")
NAMES = ("input_weights", "hidden_biases", "output_weights", "output_bias")


def test_scalar_forward_gradient_and_one_epoch():
    model, scalar = XorMlp(seed=7), ScalarMlp(seed=7)
    assert model.parameters == scalar.parameters
    for x, _ in DATA:
        assert model.logit(x) == pytest.approx(scalar.logit(x), abs=1e-14)
    analytic, reference = model.gradient(DATA), scalar.gradient(DATA)
    for name in NAMES:
        np.testing.assert_allclose(getattr(analytic, name), getattr(reference, name), atol=1e-14)
    model.fit(DATA, max_epochs=1)
    scalar.fit(DATA, max_epochs=1)
    for name in NAMES:
        np.testing.assert_allclose(getattr(model, name), getattr(scalar, name), atol=1e-14)


def test_independent_central_differences_and_parameter_copy():
    model = XorMlp(seed=13)
    snapshot = model.parameters
    assert gradient_check(model, DATA) < 1e-8
    assert model.parameters == snapshot
    model.set_parameters(snapshot)
    model.input_weights[0, 0] += 1
    assert model.parameters != snapshot
    with pytest.raises(ValueError):
        model.set_parameters(MlpParameters(((0,),), (0,), (0,), 0))


def test_extreme_logits_and_convergence():
    model = XorMlp(seed=7)
    for bias in (-1000, 1000):
        model.output_bias = bias
        assert np.isfinite(model.binary_cross_entropy(DATA))
        assert 0 <= model.predict_probability([0, 0]) <= 1
    model = XorMlp(seed=7)
    result = model.fit(DATA)
    assert result.converged and model.accuracy(DATA) == 1
    assert result.epochs == ScalarMlp(seed=7).fit(DATA).epochs
    assert not XorMlp(hidden_units=1).fit(DATA, max_epochs=10).converged


@pytest.mark.parametrize("data", [(), (((1,), 0),), (((1, np.inf), 1),), (((1, 2), 0.3),)])
def test_invalid_data(data):
    with pytest.raises(ValueError):
        XorMlp().gradient(data)


def test_invalid_controls():
    for kwargs in ({"hidden_units": 0}, {"learning_rate": np.nan}, {"feature_count": 0}):
        with pytest.raises(ValueError):
            XorMlp(**kwargs)
    for kwargs in ({"max_epochs": 0}, {"target_loss": np.inf}):
        with pytest.raises(ValueError):
            XorMlp().fit(DATA, **kwargs)
    with pytest.raises(ValueError):
        XorMlp().numerical_gradient(DATA, epsilon=0)


def test_width_experiment(capsys):
    result, widths, error = run_experiment(seeds=1)
    assert result.converged and widths[0] == (1, 0) and error < 1e-7
    assert "width=8" in capsys.readouterr().out
