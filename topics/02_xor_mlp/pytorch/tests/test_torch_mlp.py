import pytest
import torch
from ml_scratch.pure.mlp import XorMlp as ScalarMlp
from ml_scratch.pytorch.mlp import MlpParameters, XorMlp
from ml_scratch.pytorch.mlp.experiment import gradient_check, run_experiment

from ml_scratch.pure.datasets import logic_gate_dataset

DATA = logic_gate_dataset("xor")
NAMES = ("input_weights", "hidden_biases", "output_weights", "output_bias")


def test_scalar_autograd_and_finite_difference_gradients():
    model, scalar = XorMlp(seed=13), ScalarMlp(seed=13)
    assert model.parameters == scalar.parameters
    snapshot = model.parameters
    manual, auto, reference = (
        model.gradient(DATA),
        model.autograd_gradient(DATA),
        scalar.gradient(DATA),
    )
    for name in NAMES:
        values = torch.as_tensor(getattr(manual, name), dtype=torch.float64)
        torch.testing.assert_close(
            values,
            torch.as_tensor(getattr(auto, name), dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            values,
            torch.as_tensor(getattr(reference, name), dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
    assert gradient_check(model, DATA) < 1e-8
    assert model.parameters == snapshot
    assert model.input_weights.grad is None
    model.input_weights[0, 0] += 1
    assert model.parameters != snapshot  # Immutable snapshot is not a tensor view.
    model.set_parameters(snapshot)
    assert model.parameters == snapshot


def test_updates_convergence_and_nonconvergence_match_scalar():
    model, scalar = XorMlp(seed=7), ScalarMlp(seed=7)
    result, reference = model.fit(DATA), scalar.fit(DATA)
    assert result.converged and model.accuracy(DATA) == 1
    assert result.epochs == reference.epochs
    for name in NAMES:
        torch.testing.assert_close(
            torch.as_tensor(getattr(model, name), dtype=torch.float64),
            torch.as_tensor(getattr(scalar, name), dtype=torch.float64),
        )
    assert not XorMlp(hidden_units=1).fit(DATA, max_epochs=10).converged


def test_extreme_logits_and_invalid_parameters():
    model = XorMlp()
    for score in (-1000, 1000):
        model.output_bias = score
        assert torch.isfinite(torch.tensor(model.binary_cross_entropy(DATA)))
        assert 0 <= model.predict_probability([0, 0]) <= 1
    with pytest.raises(ValueError):
        model.set_parameters(MlpParameters(((1,),), (1,), (1,), 0))
    with pytest.raises(ValueError):
        model.numerical_gradient(DATA, epsilon=0)


@pytest.mark.parametrize("data", [(), (((0,), 0),), (((0, float("inf")), 1),), (((0, 0), 3),)])
def test_invalid_inputs(data):
    with pytest.raises(ValueError):
        XorMlp().gradient(data)


def test_invalid_controls_and_width_experiment(capsys):
    with pytest.raises(ValueError):
        XorMlp(hidden_units=0)
    for kwargs in ({"max_epochs": 0}, {"target_loss": float("nan")}):
        with pytest.raises(ValueError):
            XorMlp().fit(DATA, **kwargs)
    result, widths, error = run_experiment(seeds=1)
    assert result.converged and widths[0] == (1, 0) and error < 1e-7
    assert "width=8" in capsys.readouterr().out
