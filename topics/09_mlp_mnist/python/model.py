"""The topic 07 network with class-index labels and sparse input rows, fast enough for MNIST."""

import math
import random
from itertools import repeat
from operator import add, itemgetter, mul

from ml_scratch.pure.backpropagation.model import (
    Activation,
    FeedForwardNetwork,
    Loss,
    NetworkTrainingConfig,
    NetworkTrainingResult,
    activate,
    activation_derivative,
)
from ml_scratch.pure.softmax_mnist.model import SparseDataset, sparse_dataset


def _gatherer(indices):
    """A function returning (row[i] for i in indices) as a tuple, in a single C call."""
    if len(indices) == 1:
        (index,) = indices
        return lambda row: (row[index],)
    if not indices:
        return lambda row: ()
    return itemgetter(*indices)


def _activate_all(values, activation):
    if activation is Activation.IDENTITY:
        return values
    return [activate(value, activation) for value in values]


class ClassificationNetwork(FeedForwardNetwork):
    """FeedForwardNetwork with the C++ class-index overloads, over sparse rows.

    Architecture, initialization, parameter layout, single-sample passes, and checkpoints are the
    topic 07 network's. What this class adds is the classification path that MNIST needs: datasets
    of class indices instead of one-hot targets, and a first layer that reads only nonzero inputs.

    `loss`, `accuracy`, `gradient`, and `fit` therefore take a SparseDataset, or (features, label)
    pairs, rather than (features, targets) pairs. They compute exactly what the topic 07 methods
    compute on the one-hot targets; tests compare the two.

    A zero input contributes w * 0 = 0 to every first-layer pre-activation and delta * 0 = 0 to
    every first-layer weight gradient. About 81% of MNIST pixels are zero, so reading only the
    nonzero ones cuts the dominant cost, the 784-wide first layer, by about five times. It is
    exact, not an approximation. Layers after the first are dense.
    """

    def __init__(self, layers, loss, seed=0):
        if loss is not Loss.SOFTMAX_CROSS_ENTROPY:
            raise ValueError("the class-index methods require the softmax cross-entropy loss")
        super().__init__(layers, loss, seed)

    def _prepare(self, dataset):
        if not isinstance(dataset, SparseDataset):
            dataset = sparse_dataset(dataset, self.input_size)
        if not len(dataset):
            raise ValueError("dataset must not be empty")
        if dataset.feature_count != self.input_size:
            raise ValueError("sample feature count does not match the network input size")
        if any(
            type(label) is not int or not 0 <= label < self.output_size for label in dataset.labels
        ):
            raise ValueError("dataset contains a label outside the network output size")
        return dataset

    def _sparse_forward(self, gather, values):
        """Each layer's pre-activations and outputs for one row; layer 0 reads only nonzeros."""
        pre_activations, outputs = [], []
        for index, (layer, weights, biases) in enumerate(
            zip(self.layers, self.weights, self.biases, strict=True)
        ):
            if index == 0:
                z = [sum(map(mul, gather(row), values)) + b for row, b in zip(weights, biases)]
            else:
                inputs = outputs[-1]
                z = [sum(map(mul, row, inputs)) + b for row, b in zip(weights, biases)]
            pre_activations.append(z)
            outputs.append(_activate_all(z, layer.activation))
        return pre_activations, outputs

    def _logits(self, data):
        for indices, values in zip(data.indices, data.values, strict=True):
            yield self._sparse_forward(_gatherer(indices), values)[1][-1]

    def _evaluate(self, data):
        """Summed cross-entropy and the predicted classes, in one pass over the data."""
        total, predictions = 0.0, []
        for logits, label in zip(self._logits(data), data.labels, strict=True):
            largest = max(logits)
            total += largest + math.log(sum(math.exp(z - largest) for z in logits)) - logits[label]
            # index(max) is the first maximum: ties resolve to the lowest class, as in C++.
            predictions.append(logits.index(largest))
        return total, predictions

    def loss(self, dataset):
        """Mean softmax cross-entropy, log-sum-exp(z) - z_label."""
        data = self._prepare(dataset)
        return self._evaluate(data)[0] / len(data)

    def predictions(self, dataset):
        return self._evaluate(self._prepare(dataset))[1]

    def accuracy(self, dataset):
        data = self._prepare(dataset)
        predicted = self._evaluate(data)[1]
        return sum(p == y for p, y in zip(predicted, data.labels, strict=True)) / len(data)

    def confusion_matrix(self, dataset):
        """confusion[actual][predicted]."""
        data = self._prepare(dataset)
        confusion = [[0] * self.output_size for _ in range(self.output_size)]
        for predicted, actual in zip(self._evaluate(data)[1], data.labels, strict=True):
            confusion[actual][predicted] += 1
        return confusion

    def _batch_gradient(self, data, batch):
        """Summed (not averaged) gradients over the rows in `batch`.

        Returns dense weight gradients for layers 1 and up (None for layer 0), every layer's bias
        gradient, and layer 0's weight gradient as {input index: column over hidden units}, which
        holds only the inputs that were nonzero somewhere in the batch.
        """
        count = len(self.layers)
        # W^T, column by column, for delta_prev = (W^T delta) * g'(z_prev).
        transposed = [None] + [list(zip(*weights)) for weights in self.weights[1:]]
        deltas = [[] for _ in range(count)]  # [layer][sample] -> dL/dz
        inputs = [[] for _ in range(count)]  # [layer][sample] -> layer input, layers 1 and up
        first_columns = {}
        for row in batch:
            indices, values = data.indices[row], data.values[row]
            pre_activations, outputs = self._sparse_forward(_gatherer(indices), values)
            logits = outputs[-1]
            largest = max(logits)
            exponentials = [math.exp(z - largest) for z in logits]
            total = sum(exponentials)
            # Softmax composed with cross-entropy collapses dL/dz to p - y.
            delta = [value / total for value in exponentials]
            delta[data.labels[row]] -= 1.0
            for index in range(count - 1, -1, -1):
                deltas[index].append(delta)
                if index == 0:
                    break
                inputs[index].append(outputs[index - 1])
                activation = self.layers[index - 1].activation
                delta = [
                    sum(map(mul, column, delta)) * activation_derivative(z, a, activation)
                    for column, z, a in zip(
                        transposed[index], pre_activations[index - 1], outputs[index - 1]
                    )
                ]
            # dL/dW0 = delta x^T, and only the nonzero x_i contribute: column i += x_i * delta.
            for i, x in zip(indices, values, strict=True):
                column = first_columns.get(i)
                first_columns[i] = (
                    [x * d for d in delta]
                    if column is None
                    else list(map(add, column, map(mul, delta, repeat(x))))
                )
        weight_gradients = [None]
        for index in range(1, count):
            # Summed outer products delta a^T: entry (k, j) is a dot product over the batch.
            batch_inputs = list(zip(*inputs[index]))
            weight_gradients.append(
                [
                    [sum(map(mul, unit_deltas, values)) for values in batch_inputs]
                    for unit_deltas in zip(*deltas[index])
                ]
            )
        bias_gradients = [[sum(unit) for unit in zip(*layer)] for layer in deltas]
        return weight_gradients, bias_gradients, first_columns

    def gradient(self, dataset):
        """Gradient of the mean loss with respect to every parameter, in the flat layout."""
        data = self._prepare(dataset)
        weights, biases, first_columns = self._batch_gradient(data, range(len(data)))
        first = [[0.0] * self.input_size for _ in range(self.layers[0].output_size)]
        for i, column in first_columns.items():
            for row, value in zip(first, column, strict=True):
                row[i] = value
        weights[0] = first
        flat = []
        for layer_weights, layer_biases in zip(weights, biases, strict=True):
            for row in layer_weights:
                flat.extend(row)
            flat.extend(layer_biases)
        return [value / len(data) for value in flat]

    def _descend(self, data, batch, rate):
        weights, biases, first_columns = self._batch_gradient(data, batch)
        inverse = 1.0 / len(batch)
        # An input that is zero across the whole batch has a zero gradient: its weights stay put.
        if first_columns:
            touched = list(first_columns)
            for row, gradient in zip(self.weights[0], zip(*first_columns.values()), strict=True):
                for i, g in zip(touched, gradient):
                    row[i] -= rate * (g * inverse)
        for layer_weights, layer_gradients in zip(self.weights[1:], weights[1:], strict=True):
            for row, gradient in zip(layer_weights, layer_gradients, strict=True):
                row[:] = [w - rate * (g * inverse) for w, g in zip(row, gradient)]
        for layer_biases, gradient in zip(self.biases, biases, strict=True):
            layer_biases[:] = [b - rate * (g * inverse) for b, g in zip(layer_biases, gradient)]

    def fit(self, dataset, config=None):
        """Mini-batch gradient descent; the shuffle stream is a local random.Random(seed)."""
        config = config or NetworkTrainingConfig()
        data = self._prepare(dataset)
        n = len(data)
        batch_size = n if config.batch_size == 0 else min(config.batch_size, n)
        order = list(range(n))
        shuffler = random.Random(config.seed)
        history = []
        for epoch in range(1, config.max_epochs + 1):
            if config.shuffle and batch_size < n:
                shuffler.shuffle(order)
            for begin in range(0, n, batch_size):
                self._descend(data, order[begin : begin + batch_size], config.learning_rate)
            # Logits are max-shifted, so a diverging run reaches here as inf or nan, not an error.
            epoch_loss = self._evaluate(data)[0] / n
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return NetworkTrainingResult(epoch, True, tuple(history))
        return NetworkTrainingResult(config.max_epochs, False, tuple(history))
