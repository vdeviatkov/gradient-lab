import copy
import math
import random

import pytest
from ml_scratch.pure.backpropagation import (
    Activation,
    DenseLayer,
    FeedForwardNetwork,
    Loss,
    NetworkTrainingConfig,
    check_gradient,
    check_network_gradient,
    forward_difference_gradient,
    initial_parameters,
    mt19937,
    numerical_gradient,
)

ID, S, T, R = Activation.IDENTITY, Activation.SIGMOID, Activation.TANH, Activation.RELU
MSE, BCE, SOFTMAX = Loss.MEAN_SQUARED_ERROR, Loss.BINARY_CROSS_ENTROPY, Loss.SOFTMAX_CROSS_ENTROPY
XOR = (((0.0, 0.0), (0.0,)), ((0.0, 1.0), (1.0,)), ((1.0, 0.0), (1.0,)), ((1.0, 1.0), (0.0,)))
THREE_CLASSES = (
    ((-1.0, -1.0), (1.0, 0.0, 0.0)),
    ((-0.8, -1.2), (1.0, 0.0, 0.0)),
    ((1.0, -1.0), (0.0, 1.0, 0.0)),
    ((1.2, -0.7), (0.0, 1.0, 0.0)),
    ((0.0, 1.5), (0.0, 0.0, 1.0)),
    ((-0.3, 1.1), (0.0, 0.0, 1.0)),
)


def test_mt19937_reproduces_the_cpp_standard_engine():
    # The C++ standard requires the 10000th output of a default-constructed std::mt19937
    # (seed 5489) to be 4123659995.
    engine = mt19937(5489)
    for _ in range(9999):
        engine.getrandbits(32)
    assert engine.getrandbits(32) == 4123659995
    with pytest.raises(ValueError):
        mt19937(-1)


def test_initialization_limits_layout_and_zero_biases():
    layers = (DenseLayer(4, 3, T), DenseLayer(3, 2, R))
    flat = initial_parameters(layers, 7)
    assert len(flat) == 4 * 3 + 3 + 3 * 2 + 2
    glorot, he = math.sqrt(6 / 7), math.sqrt(6 / 3)
    assert all(abs(w) <= glorot for w in flat[:12]) and flat[12:15] == [0.0] * 3
    assert all(abs(w) <= he for w in flat[15:21]) and flat[21:] == [0.0] * 2
    assert max(abs(w) for w in flat[15:21]) > glorot  # He's wider limit is actually used.
    assert flat == initial_parameters(layers, 7) != initial_parameters(layers, 8)


def test_forward_pass_layout_and_round_trip_by_hand():
    network = FeedForwardNetwork((DenseLayer(2, 1),), MSE)
    assert network.parameter_count == 3
    network.set_parameters([2.0, -3.0, 0.5])
    assert network.forward([1.0, 1.0]) == pytest.approx([-0.5])
    deep = FeedForwardNetwork((DenseLayer(1, 2, T), DenseLayer(2, 1)), MSE)
    deep.set_parameters([1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.25])
    assert deep.forward([1.0]) == pytest.approx([math.tanh(1) + math.tanh(-1) + 0.25])
    values = deep.parameters()
    deep.set_parameters(values)
    assert deep.parameters() == values


def test_losses_by_hand_and_saturation():
    mse = FeedForwardNetwork((DenseLayer(1, 1),), MSE)
    mse.set_parameters([0.0, 0.0])
    assert mse.loss([((1.0,), (2.0,))]) == pytest.approx(4.0)
    bce = FeedForwardNetwork((DenseLayer(1, 1),), BCE)
    bce.set_parameters([0.0, 0.0])
    assert bce.loss([((1.0,), (1.0,))]) == pytest.approx(math.log(2))
    softmax = FeedForwardNetwork((DenseLayer(1, 3),), SOFTMAX)
    softmax.set_parameters([0.0] * 6)
    assert softmax.loss([((1.0,), (1.0, 0.0, 0.0))]) == pytest.approx(math.log(3))
    bce.set_parameters([1000.0, 0.0])
    loss = bce.loss([((1.0,), (0.0,))])
    assert math.isfinite(loss) and loss == pytest.approx(1000.0)
    assert bce.predict([1.0]) == pytest.approx([1.0])
    softmax.set_parameters([1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    assert sum(softmax.predict([1.0])) == pytest.approx(1.0)


def test_analytic_gradients_by_hand():
    # One linear unit under squared error: dL/dw = 2(z - y)x and dL/db = 2(z - y).
    network = FeedForwardNetwork((DenseLayer(2, 1),), MSE)
    network.set_parameters([0.0, 0.0, 0.0])
    assert network.gradient([((3.0, -1.0), (2.0,))]) == pytest.approx([-12.0, 4.0, -4.0])
    # Sigmoid folded into binary cross-entropy collapses the output delta to p - y.
    logistic = FeedForwardNetwork((DenseLayer(1, 1),), BCE)
    logistic.set_parameters([0.0, 0.0])
    assert logistic.gradient([((2.0,), (1.0,))]) == pytest.approx([-1.0, -0.5])


@pytest.mark.parametrize(
    "layers,loss,data",
    [
        ((DenseLayer(2, 3),), MSE, THREE_CLASSES[:3]),
        ((DenseLayer(2, 5, T), DenseLayer(5, 3, S)), MSE, THREE_CLASSES),
        ((DenseLayer(2, 4, S), DenseLayer(4, 1)), BCE, XOR),
        ((DenseLayer(2, 4, T), DenseLayer(4, 3, T), DenseLayer(3, 3)), SOFTMAX, THREE_CLASSES),
        ((DenseLayer(2, 6, R), DenseLayer(6, 3)), SOFTMAX, THREE_CLASSES),
    ],
)
def test_backpropagation_matches_central_differences(layers, loss, data):
    network = FeedForwardNetwork(layers, loss, seed=3)
    before = network.parameters()
    result = check_network_gradient(network, data)
    assert result.passed and result.relative_error < 1e-8
    assert network.parameters() == before  # The check perturbs a copy.


def test_soft_targets_keep_softmax_gradient_exact():
    # The softmax delta scales p by the target total, so targets need not sum to one.
    network = FeedForwardNetwork((DenseLayer(2, 4, T), DenseLayer(4, 3)), SOFTMAX, seed=5)
    data = [((0.2, -0.4), (0.5, 1.5, 0.0)), ((1.0, 0.3), (0.2, 0.2, 0.2))]
    assert check_network_gradient(network, data).relative_error < 1e-8


def test_central_differences_are_second_order_and_forward_first_order():
    network = FeedForwardNetwork((DenseLayer(2, 3, T), DenseLayer(3, 3)), SOFTMAX, seed=1)
    analytic = network.gradient(THREE_CLASSES)

    def error(estimate, epsilon):
        return check_gradient(analytic, estimate(network, THREE_CLASSES, epsilon)).relative_error

    # One decade of epsilon in the truncation regime: about 100x for central, 10x for one-sided.
    assert error(numerical_gradient, 1e-2) / error(numerical_gradient, 1e-3) == pytest.approx(
        100, rel=0.05
    )
    forward_ratio = error(forward_difference_gradient, 1e-2) / error(
        forward_difference_gradient, 1e-3
    )
    assert forward_ratio == pytest.approx(10, rel=0.05)


def test_check_gradient_reports_and_catches_a_corrupted_entry():
    network = FeedForwardNetwork((DenseLayer(2, 4, T), DenseLayer(4, 3)), SOFTMAX, seed=2)
    analytic = network.gradient(THREE_CLASSES)
    corrupted = list(analytic)
    corrupted[5] += 1e-3 * max(map(abs, analytic))
    result = check_gradient(corrupted, numerical_gradient(network, THREE_CLASSES))
    assert not result.passed and result.worst_parameter == 5
    assert result.analytic_value == corrupted[5]
    exact = check_gradient([0.0, 0.0], [0.0, 0.0])
    assert exact.passed and exact.relative_error == 0
    # A near-zero entry inflates the elementwise figure but not the norm ratio.
    near_zero = check_gradient([1.0, 1e-7], [1.0, 2e-7])
    assert near_zero.passed and near_zero.max_elementwise_relative_error > 0.3
    for bad in (([], []), ([1.0], [1.0, 2.0])):
        with pytest.raises(ValueError):
            check_gradient(*bad)
    with pytest.raises(ValueError):
        check_gradient([1.0], [1.0], tolerance=0)
    with pytest.raises(ValueError):
        numerical_gradient(network, THREE_CLASSES, epsilon=-1)


def test_training_learns_xor_and_gradient_still_checks():
    network = FeedForwardNetwork((DenseLayer(2, 4, T), DenseLayer(4, 1)), BCE, seed=20260910)
    config = NetworkTrainingConfig(learning_rate=0.5, max_epochs=20000, target_loss=0.02)
    result = network.fit(XOR, config)
    assert result.converged and result.epochs == len(result.loss_per_epoch)
    assert result.loss_per_epoch[-1] <= 0.02
    assert [network.predict(x)[0] >= 0.5 for x, _ in XOR] == [y[0] == 1 for _, y in XOR]
    assert check_network_gradient(network, XOR).passed


def test_training_learns_three_classes_with_mini_batches():
    network = FeedForwardNetwork((DenseLayer(2, 6, T), DenseLayer(6, 3)), SOFTMAX, seed=11)
    config = NetworkTrainingConfig(learning_rate=0.2, max_epochs=5000, batch_size=2, seed=4)
    result = network.fit(THREE_CLASSES, config)
    assert result.loss_per_epoch[-1] < result.loss_per_epoch[0]
    assert network.accuracy(THREE_CLASSES) == 1
    assert sum(network.predict([0.0, 1.5])) == pytest.approx(1.0)
    assert network.predict_class([0.0, 1.5]) == 2


def test_seed_reproducibility_and_copy_independence():
    layers = (DenseLayer(2, 4, T), DenseLayer(4, 1))
    first, second = FeedForwardNetwork(layers, BCE, 99), FeedForwardNetwork(layers, BCE, 99)
    assert first.parameters() == second.parameters()
    assert first.parameters() != FeedForwardNetwork(layers, BCE, 100).parameters()
    config = NetworkTrainingConfig(max_epochs=50, batch_size=2, seed=5)
    assert first.fit(XOR, config) == second.fit(XOR, config)
    assert first.parameters() == second.parameters()
    clone = copy.deepcopy(first)
    clone.set_parameters([0.0] * clone.parameter_count)
    assert first.parameters() == second.parameters()


def test_divergence_is_reported():
    network = FeedForwardNetwork((DenseLayer(1, 1),), MSE, seed=1)
    with pytest.raises(ArithmeticError):
        network.fit([((1e3,), (1.0,))], NetworkTrainingConfig(learning_rate=10, max_epochs=200))


def test_invalid_architectures_configs_and_data():
    for build in (
        lambda: FeedForwardNetwork((), MSE),
        lambda: FeedForwardNetwork((DenseLayer(2, 3), DenseLayer(2, 1)), MSE),
        lambda: FeedForwardNetwork((DenseLayer(2, 1, S),), BCE),
        lambda: FeedForwardNetwork((DenseLayer(2, 1),), SOFTMAX),
        lambda: FeedForwardNetwork((DenseLayer(2, 1),), "mse"),
        lambda: FeedForwardNetwork((DenseLayer(2, 1),), MSE, seed=-1),
        lambda: DenseLayer(0, 1),
        lambda: DenseLayer(1, 1.5),
        lambda: DenseLayer(1, 1, "tanh"),
    ):
        with pytest.raises(ValueError):
            build()
    for kwargs in (
        {"learning_rate": 0},
        {"learning_rate": math.inf},
        {"max_epochs": 0},
        {"batch_size": -1},
        {"seed": 2**32},
        {"target_loss": -1},
    ):
        with pytest.raises(ValueError):
            NetworkTrainingConfig(**kwargs)
    network = FeedForwardNetwork((DenseLayer(2, 1),), MSE)
    for data in ([], [((1.0,), (1.0,))], [((1.0, 2.0), (1.0, 2.0))], [((math.nan, 0.0), (1.0,))]):
        with pytest.raises(ValueError):
            network.loss(data)
    for values in ([0.0], [0.0, math.inf, 0.0]):
        with pytest.raises(ValueError):
            network.set_parameters(values)
    with pytest.raises(ValueError):
        network.forward([1.0])


def test_random_architectures_check_out():
    rng = random.Random(8)
    activations = (ID, S, T)
    for _ in range(5):
        sizes = [2] + [rng.randrange(1, 5) for _ in range(rng.randrange(1, 4))] + [3]
        layers = tuple(
            DenseLayer(a, b, rng.choice(activations)) for a, b in zip(sizes, sizes[1:-1])
        ) + (DenseLayer(sizes[-2], 3),)
        network = FeedForwardNetwork(layers, SOFTMAX, seed=rng.randrange(2**32))
        assert check_network_gradient(network, THREE_CLASSES).relative_error < 1e-8
