"""A general fully connected network trained with explicitly derived backpropagation."""

import math
import random
from dataclasses import dataclass
from enum import Enum

from ml_scratch.pure.numerics import sigmoid, softplus


class Activation(Enum):
    IDENTITY = "identity"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    # Not differentiable at zero; see the gradient-checking caveat in idea.md.
    RELU = "relu"


class Loss(Enum):
    # Mean over outputs of (prediction - target)^2, then mean over samples.
    MEAN_SQUARED_ERROR = "mean_squared_error"
    # Sigmoid applied inside the loss, so the final layer must produce raw logits.
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    # Softmax applied inside the loss, so the final layer must produce raw logits.
    SOFTMAX_CROSS_ENTROPY = "softmax_cross_entropy"


def activate(value, activation):
    if activation is Activation.SIGMOID:
        return sigmoid(value)
    if activation is Activation.TANH:
        return math.tanh(value)
    if activation is Activation.RELU:
        return max(value, 0.0)
    return value


def activation_derivative(pre_activation, value, activation):
    """Slope written through the activation value where that is cheaper and better conditioned."""
    if activation is Activation.SIGMOID:
        return value * (1.0 - value)
    if activation is Activation.TANH:
        return 1.0 - value * value
    if activation is Activation.RELU:
        # The kink at zero has no derivative; the subgradient zero is the usual convention.
        return 1.0 if pre_activation > 0.0 else 0.0
    return 1.0


def softmax(logits):
    largest = max(logits)
    exponentials = [math.exp(logit - largest) for logit in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _is_size(value):
    return type(value) is int and value > 0


@dataclass(frozen=True)
class DenseLayer:
    input_size: int
    output_size: int
    activation: Activation = Activation.IDENTITY

    def __post_init__(self):
        if not _is_size(self.input_size) or not _is_size(self.output_size):
            raise ValueError("layer sizes must be positive integers")
        if not isinstance(self.activation, Activation):
            raise ValueError("unknown activation")


@dataclass(frozen=True)
class NetworkTrainingConfig:
    learning_rate: float = 0.1
    max_epochs: int = 1000
    # Zero means one full-batch update per epoch.
    batch_size: int = 0
    shuffle: bool = True
    seed: int = 0
    # Stop once the epoch loss reaches this value; zero disables the check.
    target_loss: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not _is_size(self.max_epochs):
            raise ValueError("max_epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        if not math.isfinite(self.target_loss) or self.target_loss < 0:
            raise ValueError("target_loss must be finite and non-negative")


@dataclass(frozen=True)
class NetworkTrainingResult:
    epochs: int
    converged: bool
    # Full-dataset loss measured after each epoch's updates.
    loss_per_epoch: tuple[float, ...]


def mt19937(seed):
    """A Random whose getrandbits(32) reproduces C++ std::mt19937{seed} word for word.

    Python seeds MT19937 with init_by_array, C++ with init_genrand, so the state is built here
    with init_genrand's recurrence and installed directly. The generator itself is the same.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    state = [seed]
    for index in range(1, 624):
        previous = state[-1]
        state.append((1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF)
    engine = random.Random()
    # Position 624 means the first draw regenerates the whole block, as in a fresh C++ engine.
    engine.setstate((3, (*state, 624), None))
    return engine


def initial_parameters(layers, seed):
    """Flat Glorot or He uniform weights and zero biases, drawn exactly as the C++ network does.

    Glorot's limit sqrt(6 / (fan_in + fan_out)) balances forward and backward variance for
    activations symmetric about zero. ReLU zeroes half its inputs, so He's sqrt(6 / fan_in) is the
    factor-of-two correction.
    """
    engine = mt19937(seed)
    flat = []
    for layer in layers:
        fan_in, fan_out = layer.input_size, layer.output_size
        if layer.activation is Activation.RELU:
            limit = math.sqrt(6.0 / fan_in)
        else:
            limit = math.sqrt(6.0 / (fan_in + fan_out))
        for _ in range(fan_in * fan_out):
            flat.append(-limit + 2.0 * limit * (engine.getrandbits(32) / 4294967296.0))
        flat.extend([0.0] * fan_out)
    return flat


def validate_architecture(layers, loss):
    layers = tuple(layers)
    if not layers or not all(isinstance(layer, DenseLayer) for layer in layers):
        raise ValueError("a network needs at least one DenseLayer")
    for previous, layer in zip(layers, layers[1:]):
        if layer.input_size != previous.output_size:
            raise ValueError("layer sizes do not connect")
    if not isinstance(loss, Loss):
        raise ValueError("unknown loss")
    if loss is not Loss.MEAN_SQUARED_ERROR and layers[-1].activation is not Activation.IDENTITY:
        raise ValueError(
            "cross-entropy losses apply their own output transform, so the final layer must be "
            "linear"
        )
    if loss is Loss.SOFTMAX_CROSS_ENTROPY and layers[-1].output_size < 2:
        raise ValueError("softmax cross-entropy needs at least two outputs")
    return layers


class FeedForwardNetwork:
    """Dense layers z = W a + b, a' = g(z), with gradients derived by the chain rule.

    Parameters are also exposed as one flat vector, so the gradient check can treat any
    architecture as a single function of R^P. The layout is layer by layer, and within a layer the
    weight matrix row by row (all weights feeding output unit 0, then unit 1, ...) followed by the
    biases. It matches the C++ layout.
    """

    def __init__(self, layers, loss, seed=0):
        self.layers = validate_architecture(layers, loss)
        self.loss_function = loss
        self.weights = [
            [[0.0] * layer.input_size for _ in range(layer.output_size)] for layer in self.layers
        ]
        self.biases = [[0.0] * layer.output_size for layer in self.layers]
        self.set_parameters(initial_parameters(self.layers, seed))

    @property
    def parameter_count(self):
        return sum(layer.output_size * (layer.input_size + 1) for layer in self.layers)

    @property
    def input_size(self):
        return self.layers[0].input_size

    @property
    def output_size(self):
        return self.layers[-1].output_size

    def parameters(self):
        flat = []
        for weights, biases in zip(self.weights, self.biases, strict=True):
            for row in weights:
                flat.extend(row)
            flat.extend(biases)
        return flat

    def set_parameters(self, values):
        values = [float(value) for value in values]
        if len(values) != self.parameter_count:
            raise ValueError("parameter vector has the wrong size")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("parameters must be finite")
        cursor = 0
        for weights, biases in zip(self.weights, self.biases, strict=True):
            for row in weights:
                row[:] = values[cursor : cursor + len(row)]
                cursor += len(row)
            biases[:] = values[cursor : cursor + len(biases)]
            cursor += len(biases)

    def save(self, path):
        """Write a text checkpoint in the C++ format; see checkpoint.py."""
        from .checkpoint import save_checkpoint  # checkpoint.py imports this module.

        save_checkpoint(path, self.layers, self.loss_function, self.parameters())

    @classmethod
    def load(cls, path):
        """Rebuild a network, of this class, from a Python or C++ checkpoint."""
        from .checkpoint import load_checkpoint

        layers, loss, values = load_checkpoint(path)
        network = cls(layers, loss)
        network.set_parameters(values)
        return network

    def _vector(self, features):
        row = [float(value) for value in features]
        if len(row) != self.input_size or not all(math.isfinite(value) for value in row):
            raise ValueError(f"expected {self.input_size} finite features")
        return row

    def _validate(self, dataset):
        examples = [(self._vector(x), [float(t) for t in y]) for x, y in dataset]
        if not examples:
            raise ValueError("dataset must not be empty")
        for _, targets in examples:
            if len(targets) != self.output_size or not all(map(math.isfinite, targets)):
                raise ValueError(f"expected {self.output_size} finite targets")
        return examples

    def _forward_cache(self, features):
        """Every layer's input activation and pre-activation; the backward pass needs both."""
        activations, pre_activations = [features], []
        for layer, weights, biases in zip(self.layers, self.weights, self.biases, strict=True):
            inputs = activations[-1]
            z = [
                sum(w * a for w, a in zip(row, inputs, strict=True)) + b
                for row, b in zip(weights, biases, strict=True)
            ]
            pre_activations.append(z)
            activations.append([activate(value, layer.activation) for value in z])
        return activations, pre_activations

    def forward(self, features):
        """Raw final-layer values; for the cross-entropy losses these are logits."""
        return self._forward_cache(self._vector(features))[0][-1]

    def predict(self, features):
        """Apply the loss's output transform: softmax, sigmoid, or identity."""
        outputs = self.forward(features)
        if self.loss_function is Loss.SOFTMAX_CROSS_ENTROPY:
            return softmax(outputs)
        if self.loss_function is Loss.BINARY_CROSS_ENTROPY:
            return [sigmoid(value) for value in outputs]
        return outputs

    def predict_class(self, features):
        """Index of the largest raw output; ties resolve to the lowest index."""
        outputs = self.forward(features)
        return max(range(len(outputs)), key=outputs.__getitem__)

    def _sample_loss(self, outputs, targets):
        if self.loss_function is Loss.MEAN_SQUARED_ERROR:
            return sum((a - y) * (a - y) for a, y in zip(outputs, targets, strict=True)) / len(
                outputs
            )
        if self.loss_function is Loss.BINARY_CROSS_ENTROPY:
            # softplus(z) - y z is -log p written so that neither branch overflows.
            total = sum(softplus(z) - y * z for z, y in zip(outputs, targets, strict=True))
            return total / len(outputs)
        largest = max(outputs)
        log_normalizer = largest + math.log(sum(math.exp(z - largest) for z in outputs))
        return sum(y * (log_normalizer - z) for z, y in zip(outputs, targets, strict=True))

    def _output_delta(self, pre_activations, outputs, targets):
        """dL/dz for the final layer, which the loss and its output transform determine together."""
        m = len(outputs)
        if self.loss_function is Loss.MEAN_SQUARED_ERROR:
            # The outputs are activations here, so the chain rule still needs the activation slope.
            activation = self.layers[-1].activation
            return [
                2.0 * (a - y) / m * activation_derivative(z, a, activation)
                for z, a, y in zip(pre_activations, outputs, targets, strict=True)
            ]
        if self.loss_function is Loss.BINARY_CROSS_ENTROPY:
            # Composing sigmoid with the loss collapses the slope factor to p - y.
            return [(sigmoid(z) - y) / m for z, y in zip(outputs, targets, strict=True)]
        # Scaling p by the target total keeps the delta exact for targets that do not sum to one.
        target_total = sum(targets)
        return [target_total * p - y for p, y in zip(softmax(outputs), targets, strict=True)]

    def _accumulate(self, features, targets, scale, flat):
        """One sample's backward pass, adding scale * dL/dtheta into the flat gradient."""
        activations, pre_activations = self._forward_cache(features)
        delta = self._output_delta(pre_activations[-1], activations[-1], targets)
        offsets = self._offsets()
        for index in range(len(self.layers) - 1, -1, -1):
            layer, inputs = self.layers[index], activations[index]
            cursor = offsets[index]
            # dL/dW = delta a^T and dL/db = delta.
            for unit_delta in delta:
                for a in inputs:
                    flat[cursor] += scale * unit_delta * a
                    cursor += 1
            for unit_delta in delta:
                flat[cursor] += scale * unit_delta
                cursor += 1
            if index == 0:
                break
            # delta_prev = (W^T delta) * g'(z_prev).
            previous = self.layers[index - 1]
            weights = self.weights[index]
            delta = [
                sum(weights[output][input] * delta[output] for output in range(layer.output_size))
                * activation_derivative(
                    pre_activations[index - 1][input], inputs[input], previous.activation
                )
                for input in range(layer.input_size)
            ]

    def _subtract(self, rate, flat):
        # Updates in place without set_parameters' finiteness check, so a diverging run is
        # reported by the epoch-loss check below rather than as an invalid parameter vector.
        cursor = 0
        for weights, biases in zip(self.weights, self.biases, strict=True):
            for row in weights:
                for input in range(len(row)):
                    row[input] -= rate * flat[cursor]
                    cursor += 1
            for output in range(len(biases)):
                biases[output] -= rate * flat[cursor]
                cursor += 1

    def _offsets(self):
        offsets, total = [], 0
        for layer in self.layers:
            offsets.append(total)
            total += layer.output_size * (layer.input_size + 1)
        return offsets

    def loss(self, dataset):
        examples = self._validate(dataset)
        return sum(self._sample_loss(self._forward_cache(x)[0][-1], y) for x, y in examples) / len(
            examples
        )

    def accuracy(self, dataset):
        """Fraction of samples whose predicted class is the target's (first) largest entry."""
        examples = self._validate(dataset)
        correct = sum(
            self.predict_class(x) == max(range(len(y)), key=y.__getitem__) for x, y in examples
        )
        return correct / len(examples)

    def gradient(self, dataset):
        """Gradient of the mean dataset loss with respect to every parameter, flat layout."""
        examples = self._validate(dataset)
        flat = [0.0] * self.parameter_count
        for x, y in examples:
            self._accumulate(x, y, 1.0 / len(examples), flat)
        return flat

    def fit(self, dataset, config=None):
        """Mini-batch gradient descent; the shuffle stream is a local random.Random(seed)."""
        config = config or NetworkTrainingConfig()
        examples = self._validate(dataset)
        n = len(examples)
        batch_size = n if config.batch_size == 0 else min(config.batch_size, n)
        order = list(range(n))
        shuffler = random.Random(config.seed)
        history = []
        for epoch in range(1, config.max_epochs + 1):
            if config.shuffle and batch_size < n:
                shuffler.shuffle(order)
            for begin in range(0, n, batch_size):
                batch = order[begin : begin + batch_size]
                flat = [0.0] * self.parameter_count
                for i in batch:
                    self._accumulate(*examples[i], 1.0 / len(batch), flat)
                self._subtract(config.learning_rate, flat)
            epoch_loss = self.loss(examples)
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return NetworkTrainingResult(epoch, True, tuple(history))
        return NetworkTrainingResult(config.max_epochs, False, tuple(history))
