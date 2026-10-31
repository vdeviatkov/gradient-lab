"""Backpropagation over a whole batch at once: every layer's chain rule is one matrix product."""

import math
import random

from ml_scratch.pure.backpropagation.checkpoint import load_checkpoint, save_checkpoint
from ml_scratch.pure.backpropagation.model import (
    Activation,
    Loss,
    NetworkTrainingConfig,
    NetworkTrainingResult,
    initial_parameters,
    validate_architecture,
)

import numpy as np
from ml_scratch.numpy.validation import matrix, sigmoid, vector


def activate(z, activation):
    if activation is Activation.SIGMOID:
        return sigmoid(z)
    if activation is Activation.TANH:
        return np.tanh(z)
    if activation is Activation.RELU:
        return np.maximum(z, 0.0)
    return z


def activation_derivative(z, a, activation):
    """Slope written through the activation value where that is cheaper and better conditioned."""
    if activation is Activation.SIGMOID:
        return a * (1.0 - a)
    if activation is Activation.TANH:
        return 1.0 - a * a
    if activation is Activation.RELU:
        # The kink at zero takes the subgradient zero.
        return (z > 0.0).astype(np.float64)
    return np.ones_like(z)


def softmax(logits):
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


class FeedForwardNetwork:
    """Dense layers Z = A W^T + b over a batch of rows, with hand-derived batched gradients.

    All parameters live in one flat float64 vector, in the same layout as the scalar model. Each
    layer's weight matrix (out, in) and bias vector are reshaped views into it, so the flat vector
    the gradient check perturbs is the storage the forward pass reads, not a copy of it.
    """

    def __init__(self, layers, loss, seed=0):
        self.layers = validate_architecture(layers, loss)
        self.loss_function = loss
        self._flat = np.array(initial_parameters(self.layers, seed), dtype=np.float64)
        self.weights, self.biases = self._views(self._flat)

    def __deepcopy__(self, memo):
        # A generic deep copy would turn the layer views into independent arrays, detaching
        # them from the copied flat vector; rebuild them over the copy's own storage instead.
        clone = object.__new__(type(self))
        clone.layers, clone.loss_function = self.layers, self.loss_function
        clone._flat = self._flat.copy()
        clone.weights, clone.biases = clone._views(clone._flat)
        return clone

    def _views(self, flat):
        weights, biases, cursor = [], [], 0
        for layer in self.layers:
            size = layer.output_size * layer.input_size
            weights.append(
                flat[cursor : cursor + size].reshape(layer.output_size, layer.input_size)
            )
            biases.append(flat[cursor + size : cursor + size + layer.output_size])
            cursor += size + layer.output_size
        return weights, biases

    @property
    def parameter_count(self):
        return len(self._flat)

    @property
    def input_size(self):
        return self.layers[0].input_size

    @property
    def output_size(self):
        return self.layers[-1].output_size

    def parameters(self):
        return self._flat.copy()

    def set_parameters(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self._flat.shape:
            raise ValueError("parameter vector has the wrong size")
        if not np.isfinite(values).all():
            raise ValueError("parameters must be finite")
        self._flat[:] = values  # In place, so the layer views stay attached.

    def save(self, path):
        """Write a text checkpoint in the C++ format, shared with the scalar model."""
        save_checkpoint(path, self.layers, self.loss_function, self._flat.tolist())

    @classmethod
    def load(cls, path):
        """Rebuild a network, of this class, from a Python or C++ checkpoint."""
        layers, loss, values = load_checkpoint(path)
        network = cls(layers, loss)
        network.set_parameters(values)
        return network

    def _validate(self, dataset):
        examples = list(dataset)
        if not examples:
            raise ValueError("dataset must not be empty")
        x = matrix([features for features, _ in examples], self.input_size)
        y = np.asarray([targets for _, targets in examples], dtype=np.float64)
        if y.shape != (len(x), self.output_size) or not np.isfinite(y).all():
            raise ValueError(f"expected {self.output_size} finite targets per sample")
        return x, y

    def _forward_cache(self, x):
        activations, pre_activations = [x], []
        for layer, weights, biases in zip(self.layers, self.weights, self.biases, strict=True):
            z = activations[-1] @ weights.T + biases
            pre_activations.append(z)
            activations.append(activate(z, layer.activation))
        return activations, pre_activations

    def forward(self, features):
        """Raw final-layer values for one sample; for cross-entropy losses these are logits."""
        return self._forward_cache(vector(features, self.input_size)[None, :])[0][-1][0]

    def forward_many(self, features):
        return self._forward_cache(matrix(features, self.input_size))[0][-1]

    def predict(self, features):
        outputs = self.forward(features)
        if self.loss_function is Loss.SOFTMAX_CROSS_ENTROPY:
            return softmax(outputs)
        if self.loss_function is Loss.BINARY_CROSS_ENTROPY:
            return sigmoid(outputs)
        return outputs

    def predict_class(self, features):
        # argmax returns the first maximum, so ties resolve to the lowest index.
        return int(np.argmax(self.forward(features)))

    def _sample_losses(self, outputs, targets):
        if self.loss_function is Loss.MEAN_SQUARED_ERROR:
            residual = outputs - targets
            return (residual * residual).mean(axis=1)
        if self.loss_function is Loss.BINARY_CROSS_ENTROPY:
            # softplus(z) - y z is -log p written so that neither branch overflows.
            return (np.logaddexp(0.0, outputs) - targets * outputs).mean(axis=1)
        largest = outputs.max(axis=1, keepdims=True)
        log_normalizer = largest + np.log(np.exp(outputs - largest).sum(axis=1, keepdims=True))
        return (targets * (log_normalizer - outputs)).sum(axis=1)

    def _output_delta(self, z, a, targets):
        """dL/dz for every sample's final layer, shape (n, outputs)."""
        m = a.shape[1]
        if self.loss_function is Loss.MEAN_SQUARED_ERROR:
            slope = activation_derivative(z, a, self.layers[-1].activation)
            return 2.0 * (a - targets) / m * slope
        if self.loss_function is Loss.BINARY_CROSS_ENTROPY:
            return (sigmoid(a) - targets) / m
        return targets.sum(axis=1, keepdims=True) * softmax(a) - targets

    def _gradient(self, x, y):
        """Mean-loss gradient over the rows of x, as one flat vector with the parameter layout."""
        activations, pre_activations = self._forward_cache(x)
        delta = self._output_delta(pre_activations[-1], activations[-1], y) / len(x)
        flat = np.zeros_like(self._flat)
        weight_gradients, bias_gradients = self._views(flat)
        for index in range(len(self.layers) - 1, -1, -1):
            # Summing the per-sample outer products delta a^T is the matrix product delta^T A.
            weight_gradients[index][:] = delta.T @ activations[index]
            bias_gradients[index][:] = delta.sum(axis=0)
            if index == 0:
                break
            # delta_prev = (delta W) * g'(Z_prev), one row per sample.
            delta = (delta @ self.weights[index]) * activation_derivative(
                pre_activations[index - 1], activations[index], self.layers[index - 1].activation
            )
        return flat

    def _loss(self, x, y):
        return float(self._sample_losses(self._forward_cache(x)[0][-1], y).mean())

    def loss(self, dataset):
        return self._loss(*self._validate(dataset))

    def accuracy(self, dataset):
        x, y = self._validate(dataset)
        predicted = self._forward_cache(x)[0][-1].argmax(axis=1)
        return float(np.mean(predicted == y.argmax(axis=1)))

    def gradient(self, dataset):
        return self._gradient(*self._validate(dataset))

    def fit(self, dataset, config=None):
        """Mini-batch gradient descent with the scalar model's local random.Random shuffle."""
        config = config or NetworkTrainingConfig()
        x, y = self._validate(dataset)
        n = len(x)
        batch_size = n if config.batch_size == 0 else min(config.batch_size, n)
        order = list(range(n))
        shuffler = random.Random(config.seed)
        history = []
        for epoch in range(1, config.max_epochs + 1):
            if config.shuffle and batch_size < n:
                shuffler.shuffle(order)
            # A diverging run overflows on its way to the non-finite check below, which reports it.
            with np.errstate(over="ignore", invalid="ignore"):
                for begin in range(0, n, batch_size):
                    batch = order[begin : begin + batch_size]
                    self._flat -= config.learning_rate * self._gradient(x[batch], y[batch])
                epoch_loss = self._loss(x, y)
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return NetworkTrainingResult(epoch, True, tuple(history))
        return NetworkTrainingResult(config.max_epochs, False, tuple(history))
