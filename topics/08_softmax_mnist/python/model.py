"""Multiclass logistic regression on sparse rows: zero features cost nothing, exactly."""

import math
import random
from dataclasses import dataclass
from operator import itemgetter, mul


@dataclass(frozen=True)
class SparseDataset:
    """Feature rows kept as their nonzero (indices, values), plus integer class labels.

    A zero feature adds w * 0 = 0 to every score and (p - y) * 0 = 0 to every weight gradient, so
    skipping it changes no result. About 81% of MNIST pixels are zero, which makes this the
    difference between hours and minutes for a standard-library implementation.
    """

    indices: tuple[tuple[int, ...], ...]
    values: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    feature_count: int

    def __len__(self):
        return len(self.labels)

    def subset(self, start, stop=None):
        part = slice(start, stop)
        return SparseDataset(
            self.indices[part], self.values[part], self.labels[part], self.feature_count
        )


def sparse_dataset(examples, feature_count=None):
    """Convert dense (features, label) pairs; every row must have the same finite width."""
    examples = list(examples)
    if not examples:
        raise ValueError("dataset must not be empty")
    width = len(examples[0][0]) if feature_count is None else feature_count
    indices, values, labels = [], [], []
    for features, label in examples:
        row = [float(value) for value in features]
        if len(row) != width or width == 0 or not all(map(math.isfinite, row)):
            raise ValueError(f"expected {width} finite features per sample")
        if type(label) is not int or label < 0:
            raise ValueError("labels must be non-negative integers")
        kept = [i for i, value in enumerate(row) if value != 0.0]
        indices.append(tuple(kept))
        values.append(tuple(row[i] for i in kept))
        labels.append(label)
    return SparseDataset(tuple(indices), tuple(values), tuple(labels), width)


@dataclass(frozen=True)
class SoftmaxTrainingConfig:
    learning_rate: float = 0.1
    max_epochs: int = 20
    # Zero means one full-dataset batch per epoch.
    batch_size: int = 64
    shuffle: bool = True
    seed: int = 0
    # Applied to the weights only, never to the biases.
    l2_regularization: float = 0.0
    target_loss: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if type(self.max_epochs) is not int or self.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        _non_negative(self.l2_regularization, "l2_regularization")
        _non_negative(self.target_loss, "target_loss")


@dataclass(frozen=True)
class SoftmaxTrainingResult:
    epochs: int
    converged: bool
    # Full-dataset objective (cross-entropy plus penalty) after each epoch's updates.
    loss_per_epoch: tuple[float, ...]


def _non_negative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def softmax(logits):
    # Subtracting the largest logit leaves the distribution unchanged and keeps exp finite.
    largest = max(logits)
    exponentials = [math.exp(z - largest) for z in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


class SoftmaxRegression:
    """One linear score per class, z_k = w_k . x + b_k, turned into p = softmax(z).

    Parameters flatten to the weights row by row (all of class 0, then class 1, ...) followed by
    the biases, the C++ layout. Zero initialization is valid: the loss is convex and each class
    row is distinguished by its own labels, so there is no symmetry to break.
    """

    def __init__(self, feature_count, class_count):
        if type(feature_count) is not int or feature_count <= 0:
            raise ValueError("feature_count must be a positive integer")
        if type(class_count) is not int or class_count < 2:
            raise ValueError("class_count must be an integer >= 2")
        self.feature_count = feature_count
        self.weights = [[0.0] * feature_count for _ in range(class_count)]
        self.biases = [0.0] * class_count

    @property
    def class_count(self):
        return len(self.biases)

    @property
    def parameter_count(self):
        return self.class_count * (self.feature_count + 1)

    def parameters(self):
        return [w for row in self.weights for w in row] + list(self.biases)

    def set_parameters(self, values):
        values = [float(value) for value in values]
        if len(values) != self.parameter_count:
            raise ValueError("parameter vector has the wrong size")
        if not all(map(math.isfinite, values)):
            raise ValueError("parameters must be finite")
        d = self.feature_count
        for k, row in enumerate(self.weights):
            row[:] = values[k * d : (k + 1) * d]
        self.biases[:] = values[self.class_count * d :]

    def _prepare(self, dataset):
        if not isinstance(dataset, SparseDataset):
            dataset = sparse_dataset(dataset, self.feature_count)
        if dataset.feature_count != self.feature_count:
            raise ValueError("sample feature count does not match the model")
        if not len(dataset):
            raise ValueError("dataset must not be empty")
        if any(label >= self.class_count for label in dataset.labels):
            raise ValueError("dataset contains a label outside the class count")
        return dataset

    def _sparse_scores(self, indices, values):
        if not indices:
            return list(self.biases)
        gather = itemgetter(*indices)
        if len(indices) == 1:
            return [b + row[indices[0]] * values[0] for row, b in zip(self.weights, self.biases)]
        return [
            b + sum(map(mul, gather(row), values))
            for row, b in zip(self.weights, self.biases, strict=True)
        ]

    def _dense(self, features):
        row = [float(value) for value in features]
        if len(row) != self.feature_count or not all(map(math.isfinite, row)):
            raise ValueError(f"expected {self.feature_count} finite features")
        indices = tuple(i for i, value in enumerate(row) if value != 0.0)
        return indices, tuple(row[i] for i in indices)

    def scores(self, features):
        return self._sparse_scores(*self._dense(features))

    def predict_probabilities(self, features):
        return softmax(self.scores(features))

    def predict(self, features):
        """Index of the largest score; ties resolve to the lowest class index."""
        logits = self.scores(features)
        return max(range(len(logits)), key=logits.__getitem__)

    def _penalty(self, l2):
        if not l2:
            return 0.0
        return 0.5 * l2 * sum(w * w for row in self.weights for w in row)

    def cross_entropy(self, dataset, l2_regularization=0.0):
        """Mean -log p_y, computed as log-sum-exp minus the true logit, plus (l2 / 2) ||W||^2."""
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        total = 0.0
        for indices, values, label in zip(data.indices, data.values, data.labels, strict=True):
            logits = self._sparse_scores(indices, values)
            largest = max(logits)
            total += largest + math.log(sum(math.exp(z - largest) for z in logits)) - logits[label]
        return total / len(data) + self._penalty(l2_regularization)

    def predictions(self, dataset):
        data = self._prepare(dataset)
        result = []
        for indices, values in zip(data.indices, data.values, strict=True):
            logits = self._sparse_scores(indices, values)
            result.append(max(range(len(logits)), key=logits.__getitem__))
        return result

    def accuracy(self, dataset):
        data = self._prepare(dataset)
        correct = sum(p == y for p, y in zip(self.predictions(data), data.labels, strict=True))
        return correct / len(data)

    def confusion_matrix(self, dataset):
        """confusion[actual][predicted] counts."""
        data = self._prepare(dataset)
        confusion = [[0] * self.class_count for _ in range(self.class_count)]
        for predicted, actual in zip(self.predictions(data), data.labels, strict=True):
            confusion[actual][predicted] += 1
        return confusion

    def _accumulate(self, data, positions, gradient_rows, bias_gradient):
        """Add sum_i (p_i - y_i) x_i^T into the gradient rows; return the touched features."""
        touched = set()
        for i in positions:
            indices, values, label = data.indices[i], data.values[i], data.labels[i]
            probabilities = softmax(self._sparse_scores(indices, values))
            probabilities[label] -= 1.0  # Softmax composed with cross-entropy collapses to p - y.
            touched.update(indices)
            for row, delta in zip(gradient_rows, probabilities, strict=True):
                for feature, value in zip(indices, values, strict=True):
                    row[feature] += delta * value
            for k, delta in enumerate(probabilities):
                bias_gradient[k] += delta
        return touched

    def gradient(self, dataset, l2_regularization=0.0):
        """Gradient of the mean objective, in the flat parameter layout."""
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        rows = [[0.0] * self.feature_count for _ in range(self.class_count)]
        biases = [0.0] * self.class_count
        self._accumulate(data, range(len(data)), rows, biases)
        n = len(data)
        flat = [
            g / n + l2_regularization * w
            for row, weights in zip(rows, self.weights, strict=True)
            for g, w in zip(row, weights, strict=True)
        ]
        return flat + [g / n for g in biases]

    def fit(self, dataset, config=None):
        """Mini-batch SGD; the shuffle stream is a local random.Random(seed)."""
        config = config or SoftmaxTrainingConfig()
        data = self._prepare(dataset)
        n = len(data)
        batch_size = n if config.batch_size == 0 else min(config.batch_size, n)
        order = list(range(n))
        shuffler = random.Random(config.seed)
        rate, l2 = config.learning_rate, config.l2_regularization
        everything = range(self.feature_count)
        history = []
        for epoch in range(1, config.max_epochs + 1):
            if config.shuffle and batch_size < n:
                shuffler.shuffle(order)
            for begin in range(0, n, batch_size):
                batch = order[begin : begin + batch_size]
                inverse = 1.0 / len(batch)
                rows = [[0.0] * self.feature_count for _ in range(self.class_count)]
                biases = [0.0] * self.class_count
                touched = self._accumulate(data, batch, rows, biases)
                # Without a penalty an untouched weight has a zero gradient and stays put; the
                # penalty decays every weight, so then the whole matrix is updated.
                features = everything if l2 else touched
                for weights, gradient in zip(self.weights, rows, strict=True):
                    for f in features:
                        weights[f] -= rate * (gradient[f] * inverse + l2 * weights[f])
                for k, gradient in enumerate(biases):
                    self.biases[k] -= rate * gradient * inverse
            epoch_loss = self.cross_entropy(data, l2)
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return SoftmaxTrainingResult(epoch, True, tuple(history))
        return SoftmaxTrainingResult(config.max_epochs, False, tuple(history))
