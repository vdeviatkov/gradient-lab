"""Softmax regression over batches of pixel rows: scores and gradients are matrix products."""

import math
import random
from dataclasses import dataclass

from ml_scratch.pure.softmax_mnist.model import SoftmaxTrainingConfig, SoftmaxTrainingResult

import numpy as np
from ml_scratch.numpy.validation import vector

# Rows per evaluation chunk: bounds the temporary float64 copy of a 54000-row byte matrix.
CHUNK = 4096


@dataclass(frozen=True)
class ByteImageDataset:
    """Images kept as their stored uint8 bytes, scaled by 1/255 one batch at a time.

    A float64 copy of 54000 MNIST images would take 339 MB. Converting each gathered batch
    instead yields exactly the same values, byte / 255.0, at 1/8 of the resident memory.
    """

    pixels: np.ndarray  # (n, features) uint8
    labels: np.ndarray  # (n,) int64

    def __post_init__(self):
        if self.pixels.dtype != np.uint8 or self.pixels.ndim != 2 or not len(self.pixels):
            raise ValueError("pixels must be a nonempty two-dimensional uint8 array")
        if self.labels.shape != (len(self.pixels),) or self.labels.dtype.kind not in "iu":
            raise ValueError("labels must be one integer per image")

    def __len__(self):
        return len(self.labels)

    @property
    def feature_count(self):
        return self.pixels.shape[1]

    def features(self, rows=slice(None)):
        return self.pixels[rows] / 255.0


def byte_images(images, labels, feature_count):
    """Build a dataset from MNIST byte strings, one per image."""
    pixels = np.frombuffer(b"".join(images), dtype=np.uint8).reshape(len(images), feature_count)
    return ByteImageDataset(pixels, np.asarray(labels, dtype=np.int64))


def _non_negative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def softmax(logits):
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


class SoftmaxRegression:
    """Z = X W^T + b for a batch of rows X, p = softmax(Z), fitted by mean cross-entropy.

    One flat float64 vector holds the parameters in the scalar layout; `weights` (classes,
    features) and `biases` are views into it.
    """

    def __init__(self, feature_count, class_count):
        for value, name, low in (
            (feature_count, "feature_count", 1),
            (class_count, "class_count", 2),
        ):
            if type(value) is not int or value < low:
                raise ValueError(f"{name} must be an integer >= {low}")
        self.feature_count = feature_count
        self._flat = np.zeros(class_count * (feature_count + 1))
        self.weights = self._flat[: class_count * feature_count].reshape(class_count, feature_count)
        self.biases = self._flat[class_count * feature_count :]

    @property
    def class_count(self):
        return len(self.biases)

    @property
    def parameter_count(self):
        return len(self._flat)

    def parameters(self):
        return self._flat.copy()

    def set_parameters(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self._flat.shape:
            raise ValueError("parameter vector has the wrong size")
        if not np.isfinite(values).all():
            raise ValueError("parameters must be finite")
        self._flat[:] = values

    def _prepare(self, dataset):
        if not isinstance(dataset, (ByteImageDataset, _DenseDataset)):
            examples = list(dataset)
            if not examples:
                raise ValueError("dataset must not be empty")
            x = np.asarray([features for features, _ in examples], dtype=np.float64)
            y = np.asarray([label for _, label in examples])
            if x.ndim != 2 or x.shape[1] != self.feature_count or not np.isfinite(x).all():
                raise ValueError(f"expected {self.feature_count} finite features per sample")
            if y.dtype.kind not in "iu" or (y < 0).any():
                raise ValueError("labels must be non-negative integers")
            dataset = _DenseDataset(x, y.astype(np.int64))
        if dataset.feature_count != self.feature_count:
            raise ValueError("sample feature count does not match the model")
        if (dataset.labels >= self.class_count).any():
            raise ValueError("dataset contains a label outside the class count")
        return dataset

    def scores(self, features):
        return vector(features, self.feature_count) @ self.weights.T + self.biases

    def predict_probabilities(self, features):
        return softmax(self.scores(features))

    def predict(self, features):
        # argmax returns the first maximum: ties resolve to the lowest class index.
        return int(np.argmax(self.scores(features)))

    def _chunks(self, data):
        for start in range(0, len(data), CHUNK):
            rows = slice(start, start + CHUNK)
            yield data.features(rows) @ self.weights.T + self.biases, data.labels[rows]

    def cross_entropy(self, dataset, l2_regularization=0.0):
        """Mean of log-sum-exp(z) - z_y, plus (l2 / 2) ||W||^2."""
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        total = 0.0
        for logits, labels in self._chunks(data):
            largest = logits.max(axis=1)
            log_normalizer = largest + np.log(np.exp(logits - largest[:, None]).sum(axis=1))
            total += float((log_normalizer - logits[np.arange(len(labels)), labels]).sum())
        penalty = 0.5 * l2_regularization * float(np.sum(self.weights * self.weights))
        return total / len(data) + penalty

    def predictions(self, dataset):
        data = self._prepare(dataset)
        return np.concatenate([logits.argmax(axis=1) for logits, _ in self._chunks(data)])

    def accuracy(self, dataset):
        data = self._prepare(dataset)
        return float(np.mean(self.predictions(data) == data.labels))

    def confusion_matrix(self, dataset):
        """confusion[actual, predicted], counted with one bincount over flattened pairs."""
        data = self._prepare(dataset)
        pairs = data.labels * self.class_count + self.predictions(data)
        return np.bincount(pairs, minlength=self.class_count**2).reshape(
            self.class_count, self.class_count
        )

    def _batch_gradient(self, x, labels):
        """Summed (not averaged) gradients: G_W = (P - Y)^T X and g_b = column sums of P - Y."""
        delta = softmax(x @ self.weights.T + self.biases)
        # Softmax composed with cross-entropy collapses dL/dz to p - y.
        delta[np.arange(len(labels)), labels] -= 1.0
        return delta.T @ x, delta.sum(axis=0)

    def gradient(self, dataset, l2_regularization=0.0):
        """Gradient of the mean objective, in the flat parameter layout."""
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        weights = np.zeros_like(self.weights)
        biases = np.zeros_like(self.biases)
        for start in range(0, len(data), CHUNK):
            rows = slice(start, start + CHUNK)
            g_w, g_b = self._batch_gradient(data.features(rows), data.labels[rows])
            weights += g_w
            biases += g_b
        weights = weights / len(data) + l2_regularization * self.weights
        return np.concatenate((weights.ravel(), biases / len(data)))

    def fit(self, dataset, config=None):
        """Mini-batch SGD with the scalar model's local random.Random shuffle stream."""
        config = config or SoftmaxTrainingConfig()
        data = self._prepare(dataset)
        n = len(data)
        batch_size = n if config.batch_size == 0 else min(config.batch_size, n)
        order = list(range(n))
        shuffler = random.Random(config.seed)
        rate, l2 = config.learning_rate, config.l2_regularization
        history = []
        for epoch in range(1, config.max_epochs + 1):
            if config.shuffle and batch_size < n:
                shuffler.shuffle(order)
            # A diverging run overflows on its way to the non-finite check below, which reports it.
            with np.errstate(over="ignore", invalid="ignore"):
                for begin in range(0, n, batch_size):
                    batch = order[begin : begin + batch_size]
                    g_w, g_b = self._batch_gradient(data.features(batch), data.labels[batch])
                    inverse = 1.0 / len(batch)
                    self.weights -= rate * (g_w * inverse + l2 * self.weights)
                    self.biases -= rate * g_b * inverse
                epoch_loss = self.cross_entropy(data, l2)
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return SoftmaxTrainingResult(epoch, True, tuple(history))
        return SoftmaxTrainingResult(config.max_epochs, False, tuple(history))


@dataclass(frozen=True)
class _DenseDataset:
    """Already-scaled float features, for small (features, label) inputs such as tests."""

    x: np.ndarray
    labels: np.ndarray

    def __len__(self):
        return len(self.labels)

    @property
    def feature_count(self):
        return self.x.shape[1]

    def features(self, rows=slice(None)):
        return self.x[rows]
