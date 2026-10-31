import copy
import math
from dataclasses import astuple

import pytest
import torch
from ml_scratch.pure.backpropagation import FeedForwardNetwork as ScalarNetwork
from ml_scratch.pure.backpropagation import check_gradient as scalar_check
from ml_scratch.pure.backpropagation.experiment import (
    BINARY_DATA,
    CONFIGURATIONS,
    REGRESSION_DATA,
    build_spiral,
)
from ml_scratch.pytorch.backpropagation import (
    Activation,
    DenseLayer,
    FeedForwardNetwork,
    Loss,
    NetworkTrainingConfig,
    check_gradient,
    check_network_gradient,
    forward_difference_gradient,
    numerical_gradient,
)
from ml_scratch.pytorch.backpropagation.experiment import run_experiment

ID, S, T, R = Activation.IDENTITY, Activation.SIGMOID, Activation.TANH, Activation.RELU
MSE, BCE, SOFTMAX = Loss.MEAN_SQUARED_ERROR, Loss.BINARY_CROSS_ENTROPY, Loss.SOFTMAX_CROSS_ENTROPY
SPIRAL = build_spiral()


def close(actual, expected, rtol=1e-13, atol=0.0):
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )


def data_for(loss):
    return {MSE: REGRESSION_DATA, BCE: BINARY_DATA}.get(loss, SPIRAL.train[:60])


@pytest.mark.parametrize("name,layers,loss", CONFIGURATIONS)
def test_batched_passes_match_scalar_and_central_differences(name, layers, loss):
    data = data_for(loss)
    network, scalar = (
        FeedForwardNetwork(layers, loss, 20260910),
        ScalarNetwork(layers, loss, 20260910),
    )
    close(network.parameters(), scalar.parameters())
    assert network.loss(data) == pytest.approx(scalar.loss(data), rel=1e-14)
    close(network.gradient(data), scalar.gradient(data), rtol=1e-12, atol=1e-16)
    close(
        network.forward_many([x for x, _ in data]), [scalar.forward(x) for x, _ in data], rtol=1e-13
    )
    for x, _ in data[:5]:
        close(network.predict(x), scalar.predict(x), rtol=1e-13)
        assert network.predict_class(x) == scalar.predict_class(x)
    assert network.accuracy(data) == scalar.accuracy(data)
    assert check_network_gradient(network, data).relative_error < 1e-8


def test_flat_storage_views_and_deep_copies():
    network = FeedForwardNetwork((DenseLayer(2, 3, T), DenseLayer(3, 2)), MSE, seed=4)
    network.set_parameters(torch.arange(network.parameter_count, dtype=torch.float64))
    close(network.weights[0], [[0, 1], [2, 3], [4, 5]])
    close(network.biases[1], [15, 16])
    parameters = network.parameters()
    parameters[0] = 99  # parameters() is a copy.
    assert float(network.weights[0][0, 0]) == 0
    clone = copy.deepcopy(network)
    clone._flat[0] = 42  # The copy's views follow its own storage, and only its own.
    assert clone.weights[0][0, 0] == 42 and network.weights[0][0, 0] == 0


def test_training_trajectory_matches_scalar():
    layers = (DenseLayer(2, 6, T), DenseLayer(6, 3))
    config = NetworkTrainingConfig(learning_rate=0.5, max_epochs=40, batch_size=32, seed=3)
    network, scalar = FeedForwardNetwork(layers, SOFTMAX, 9), ScalarNetwork(layers, SOFTMAX, 9)
    result, reference = network.fit(SPIRAL.train, config), scalar.fit(SPIRAL.train, config)
    assert result.epochs == reference.epochs and not result.converged
    close(result.loss_per_epoch, reference.loss_per_epoch, rtol=1e-11)
    close(network.parameters(), scalar.parameters(), rtol=1e-10, atol=1e-13)
    converged = FeedForwardNetwork((DenseLayer(2, 4, T), DenseLayer(4, 1)), BCE, 20260910)
    trained = converged.fit(BINARY_DATA, NetworkTrainingConfig(0.5, 20000, target_loss=0.02))
    assert trained.converged and trained.loss_per_epoch[-1] <= 0.02


def test_checker_matches_scalar_and_central_differences_are_second_order():
    network = FeedForwardNetwork((DenseLayer(2, 3, T), DenseLayer(3, 3)), SOFTMAX, seed=1)
    data = SPIRAL.train[:30]
    analytic = network.gradient(data)
    numerical = numerical_gradient(network, data)
    expected = scalar_check(analytic.tolist(), numerical.tolist())
    assert astuple(check_gradient(analytic, numerical)) == pytest.approx(astuple(expected))

    def error(estimate, epsilon):
        return check_gradient(analytic, estimate(network, data, epsilon)).relative_error

    central = error(numerical_gradient, 1e-2) / error(numerical_gradient, 1e-3)
    forward = error(forward_difference_gradient, 1e-2) / error(forward_difference_gradient, 1e-3)
    assert central == pytest.approx(100, rel=0.05) and forward == pytest.approx(10, rel=0.05)
    corrupted = analytic.clone()
    corrupted[4] *= 1.01
    result = check_gradient(corrupted, numerical)
    assert not result.passed and result.worst_parameter == 4


def test_stable_saturated_losses_and_divergence():
    bce = FeedForwardNetwork((DenseLayer(1, 1),), BCE)
    bce.set_parameters([1000.0, 0.0])
    assert bce.loss([((1.0,), (0.0,))]) == pytest.approx(1000.0)
    softmax = FeedForwardNetwork((DenseLayer(1, 3),), SOFTMAX)
    softmax.set_parameters([1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    assert math.isfinite(softmax.loss([((1.0,), (0.0, 1.0, 0.0))]))
    assert float(softmax.predict([1.0]).sum()) == pytest.approx(1.0)
    diverging = FeedForwardNetwork((DenseLayer(1, 1),), MSE, seed=1)
    with pytest.raises(ArithmeticError):
        diverging.fit([((1e3,), (1.0,))], NetworkTrainingConfig(learning_rate=10, max_epochs=200))


def test_invalid_inputs():
    network = FeedForwardNetwork((DenseLayer(2, 1),), MSE)
    for data in ([], [((1.0,), (1.0,))], [((1.0, 2.0), (1.0, 2.0))], [((math.nan, 0.0), (1.0,))]):
        with pytest.raises(ValueError):
            network.loss(data)
    for values in ([0.0], [0.0, math.inf, 0.0]):
        with pytest.raises(ValueError):
            network.set_parameters(values)
    for bad in (([], []), ([1.0], [1.0, 2.0]), ([[1.0]], [[1.0]])):
        with pytest.raises(ValueError):
            check_gradient(*bad)
    with pytest.raises(ValueError):
        numerical_gradient(network, [((0.0, 0.0), (1.0,))], epsilon=0)
    with pytest.raises(ValueError):
        FeedForwardNetwork((DenseLayer(2, 1, S),), BCE)


def test_torch_experiment_meets_criteria_and_matches_scalar_training():
    result = run_experiment()  # Raises unless every recorded criterion holds.
    assert [c.parameters for c in result.configurations] == [6, 27, 22, 17, 51, 103, 81]
    assert f"{result.step_sizes[0].central:.3e}" == "5.329e-04"
    assert result.detection.worst_parameter == 7
    # Recorded pure-Python 4000-epoch values; the shuffle stream and initialization are shared.
    assert (result.linear.train, result.linear.validation, result.linear.test) == pytest.approx(
        (139 / 270, 48 / 90, 48 / 90)
    )
    assert (result.mlp.train, result.mlp.validation, result.mlp.test) == (1, 1, 1)
    assert f"{result.trajectory[0].largest_gradient:.3e}" == "9.913e-02"
    assert not result.trajectory[2].result.passed  # The documented norm-ratio degeneracy.


@pytest.mark.parametrize("name,layers,loss", CONFIGURATIONS)
def test_manual_backpropagation_matches_independent_autograd(name, layers, loss):
    data = data_for(loss)
    network = FeedForwardNetwork(layers, loss, 20260910)
    before = network.parameters()
    autograd = network.autograd_gradient(data)
    torch.testing.assert_close(network.gradient(data), autograd, rtol=1e-12, atol=1e-16)
    assert torch.equal(network.parameters(), before)
    assert not network._flat.requires_grad and network._flat.grad is None
    assert network.gradient(data).dtype == torch.float64


def test_tensor_inputs_with_graphs_are_detached():
    x = torch.tensor([[0.2, -0.4], [1.0, 0.3]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    network = FeedForwardNetwork((DenseLayer(2, 4, T), DenseLayer(4, 3)), SOFTMAX, seed=6)
    data = list(zip(x, y, strict=True))
    gradient = network.gradient(data)
    assert not gradient.requires_grad and x.grad is None
    network.fit(data, NetworkTrainingConfig(max_epochs=3))
    assert not network._flat.requires_grad


def test_checkpoints_match_the_scalar_files_byte_for_byte(tmp_path):
    layers = (DenseLayer(2, 5, R), DenseLayer(5, 3))
    network, scalar = FeedForwardNetwork(layers, SOFTMAX, 31), ScalarNetwork(layers, SOFTMAX, 31)
    ours, theirs = tmp_path / "torch.checkpoint", tmp_path / "scalar.checkpoint"
    network.save(ours)
    scalar.save(theirs)
    assert ours.read_bytes() == theirs.read_bytes()
    network.fit(SPIRAL.train[:40], NetworkTrainingConfig(max_epochs=5, batch_size=8))
    network.save(ours)
    restored = FeedForwardNetwork.load(ours)
    assert type(restored) is FeedForwardNetwork and restored.layers == layers
    assert torch.equal(restored.parameters(), network.parameters())
    assert ScalarNetwork.load(ours).parameters() == network.parameters().tolist()
