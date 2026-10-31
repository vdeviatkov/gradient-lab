"""Softmax regression in CPU float64 tensors with hand-derived gradients and an autograd oracle."""

import math
import random
from dataclasses import dataclass

import torch
from ml_scratch.pure.softmax_mnist.model import SoftmaxTrainingConfig, SoftmaxTrainingResult

from ml_scratch.pytorch.validation import tensor, vector

# Rows per evaluation chunk: bounds the temporary float64 copy of a 54000-row byte matrix.
CHUNK = 4096
_INTEGER_TYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


@dataclass(frozen=True)
class ByteImageDataset:
    """Images kept as their stored uint8 bytes, scaled by 1/255 one batch at a time."""

    pixels: torch.Tensor  # (n, features) uint8
    labels: torch.Tensor  # (n,) int64

    def __post_init__(self):
        if self.pixels.dtype != torch.uint8 or self.pixels.ndim != 2 or not len(self.pixels):
            raise ValueError("pixels must be a nonempty two-dimensional uint8 tensor")
        if self.labels.shape != (len(self.pixels),) or self.labels.dtype not in _INTEGER_TYPES:
            raise ValueError("labels must be one integer per image")

    def __len__(self):
        return len(self.labels)

    @property
    def feature_count(self):
        return self.pixels.shape[1]

    def features(self, rows=slice(None)):
        return self.pixels[rows].to(torch.float64) / 255.0


def byte_images(images, labels, feature_count):
    """Build a dataset from MNIST byte strings, one per image."""
    pixels = torch.frombuffer(bytearray(b"".join(images)), dtype=torch.uint8)
    return ByteImageDataset(
        pixels.reshape(len(images), feature_count), torch.tensor(labels, dtype=torch.int64)
    )


@dataclass(frozen=True)
class DenseDataset:
    """Already-scaled float features, for small (features, label) inputs such as tests."""

    x: torch.Tensor
    labels: torch.Tensor

    def __len__(self):
        return len(self.labels)

    @property
    def feature_count(self):
        return self.x.shape[1]

    def features(self, rows=slice(None)):
        return self.x[rows]


def _non_negative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def softmax(logits):
    shifted = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    return shifted / shifted.sum(dim=-1, keepdim=True)


def _cross_entropy_sum(logits, labels):
    """Sum over rows of log-sum-exp(z) - z_y, with the max shift; differentiable for autograd."""
    largest = logits.max(dim=1, keepdim=True).values
    log_normalizer = largest[:, 0] + torch.log(torch.exp(logits - largest).sum(dim=1))
    return (log_normalizer - logits.gather(1, labels[:, None])[:, 0]).sum()


class SoftmaxRegression:
    """Z = X W^T + b for a batch of rows X, p = softmax(Z), fitted by mean cross-entropy.

    One flat float64 tensor holds the parameters in the scalar layout; `weights` (classes,
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
        self._flat = torch.zeros(class_count * (feature_count + 1), dtype=torch.float64)
        self.weights, self.biases = self._views(self._flat)

    def _views(self, flat):
        cut = self.feature_count * (len(flat) // (self.feature_count + 1))
        return flat[:cut].view(-1, self.feature_count), flat[cut:]

    @property
    def class_count(self):
        return len(self.biases)

    @property
    def parameter_count(self):
        return len(self._flat)

    def parameters(self):
        return self._flat.clone()

    def set_parameters(self, values):
        values = tensor(values)
        if values.shape != self._flat.shape:
            raise ValueError("parameter vector has the wrong size")
        if not torch.isfinite(values).all():
            raise ValueError("parameters must be finite")
        self._flat.copy_(values)

    def _prepare(self, dataset):
        if not isinstance(dataset, (ByteImageDataset, DenseDataset)):
            examples = list(dataset)
            if not examples:
                raise ValueError("dataset must not be empty")
            x = tensor([features for features, _ in examples])
            y = torch.as_tensor([label for _, label in examples])
            if x.ndim != 2 or x.shape[1] != self.feature_count or not torch.isfinite(x).all():
                raise ValueError(f"expected {self.feature_count} finite features per sample")
            if y.dtype not in _INTEGER_TYPES or (y < 0).any():
                raise ValueError("labels must be non-negative integers")
            dataset = DenseDataset(x, y.to(torch.int64))
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
        # Ties resolve to the lowest class index, matching the scalar model.
        return int(torch.argmax(self.scores(features)))

    def _chunks(self, data):
        for start in range(0, len(data), CHUNK):
            rows = slice(start, start + CHUNK)
            yield data.features(rows) @ self.weights.T + self.biases, data.labels[rows]

    @torch.no_grad()
    def cross_entropy(self, dataset, l2_regularization=0.0):
        """Mean of log-sum-exp(z) - z_y, plus (l2 / 2) ||W||^2."""
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        total = sum(
            float(_cross_entropy_sum(logits, labels)) for logits, labels in self._chunks(data)
        )
        penalty = 0.5 * l2_regularization * float((self.weights * self.weights).sum())
        return total / len(data) + penalty

    @torch.no_grad()
    def predictions(self, dataset):
        data = self._prepare(dataset)
        return torch.cat([logits.argmax(dim=1) for logits, _ in self._chunks(data)])

    def accuracy(self, dataset):
        data = self._prepare(dataset)
        return int((self.predictions(data) == data.labels).sum()) / len(data)

    def confusion_matrix(self, dataset):
        """confusion[actual, predicted], counted with one bincount over flattened pairs."""
        data = self._prepare(dataset)
        pairs = data.labels * self.class_count + self.predictions(data)
        k = self.class_count
        return torch.bincount(pairs, minlength=k * k).reshape(k, k)

    def _batch_gradient(self, x, labels):
        """Summed (not averaged) gradients: G_W = (P - Y)^T X and g_b = column sums of P - Y."""
        delta = softmax(x @ self.weights.T + self.biases)
        # Softmax composed with cross-entropy collapses dL/dz to p - y.
        delta[torch.arange(len(labels)), labels] -= 1.0
        return delta.T @ x, delta.sum(dim=0)

    @torch.no_grad()
    def gradient(self, dataset, l2_regularization=0.0):
        """Gradient of the mean objective, in the flat parameter layout."""
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        weights = torch.zeros_like(self.weights)
        biases = torch.zeros_like(self.biases)
        for start in range(0, len(data), CHUNK):
            rows = slice(start, start + CHUNK)
            g_w, g_b = self._batch_gradient(data.features(rows), data.labels[rows])
            weights += g_w
            biases += g_b
        weights = weights / len(data) + l2_regularization * self.weights
        return torch.cat((weights.reshape(-1), biases / len(data)))

    def autograd_gradient(self, dataset, l2_regularization=0.0):
        """An independent oracle: torch.autograd of the forward objective on a cloned leaf.

        It never calls the hand-derived gradient and leaves the parameters and .grad untouched.
        """
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        with torch.enable_grad():
            flat = self._flat.clone().requires_grad_(True)
            weights, biases = self._views(flat)
            total = sum(
                _cross_entropy_sum(data.features(rows) @ weights.T + biases, data.labels[rows])
                for rows in (slice(s, s + CHUNK) for s in range(0, len(data), CHUNK))
            )
            objective = total / len(data) + 0.5 * l2_regularization * (weights * weights).sum()
            (gradient,) = torch.autograd.grad(objective, flat)
        return gradient.detach()

    @torch.no_grad()
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
