"""The topic 07 PyTorch network with class-index labels over byte-stored MNIST images."""

import math
import random

import torch
from ml_scratch.pure.backpropagation.model import Loss, NetworkTrainingConfig, NetworkTrainingResult
from ml_scratch.pytorch.backpropagation.model import FeedForwardNetwork
from ml_scratch.pytorch.softmax_mnist.model import ByteImageDataset, DenseDataset

from ml_scratch.pytorch.validation import tensor

# Rows per evaluation chunk: bounds the float64 copy of the images and every layer's activations.
CHUNK = 4096
_INTEGER_TYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def dense_examples(examples, feature_count):
    """Small (features, label) pairs, such as test data, as an already-scaled DenseDataset."""
    examples = list(examples)
    if not examples:
        raise ValueError("dataset must not be empty")
    x = tensor([features for features, _ in examples])
    labels = torch.as_tensor([label for _, label in examples])
    if x.ndim != 2 or x.shape[1] != feature_count or not torch.isfinite(x).all():
        raise ValueError(f"expected {feature_count} finite features per sample")
    if labels.dtype not in _INTEGER_TYPES or (labels < 0).any():
        raise ValueError("labels must be non-negative integers")
    return DenseDataset(x, labels.to(torch.int64))


def _cross_entropy_sum(logits, labels):
    """Summed log-sum-exp(z) - z_label, max-shifted."""
    largest = logits.max(dim=1).values
    log_normalizer = largest + torch.log(torch.exp(logits - largest[:, None]).sum(dim=1))
    return (log_normalizer - logits[torch.arange(len(labels)), labels]).sum()


class ClassificationNetwork(FeedForwardNetwork):
    """FeedForwardNetwork with the C++ class-index overloads, over byte or dense datasets.

    The NumPy design in CPU float64 tensors: a batch's labels become one-hot rows and the topic 07
    `_gradient` does the rest. `autograd_gradient` is an independent oracle for the same quantity.
    """

    def __init__(self, layers, loss, seed=0):
        if loss is not Loss.SOFTMAX_CROSS_ENTROPY:
            raise ValueError("the class-index methods require the softmax cross-entropy loss")
        super().__init__(layers, loss, seed)

    def _prepare(self, dataset):
        if not isinstance(dataset, (ByteImageDataset, DenseDataset)):
            dataset = dense_examples(dataset, self.input_size)
        if dataset.feature_count != self.input_size:
            raise ValueError("sample feature count does not match the network input size")
        if (dataset.labels >= self.output_size).any():
            raise ValueError("dataset contains a label outside the network output size")
        return dataset

    def _one_hot(self, labels):
        targets = torch.zeros((len(labels), self.output_size), dtype=torch.float64)
        targets[torch.arange(len(labels)), labels] = 1.0
        return targets

    @torch.no_grad()
    def _evaluate(self, data):
        """Summed cross-entropy and the predicted classes, chunk by chunk."""
        total, predictions = 0.0, []
        for start in range(0, len(data), CHUNK):
            rows = slice(start, start + CHUNK)
            logits = self._forward_cache(data.features(rows))[0][-1]
            total += float(_cross_entropy_sum(logits, data.labels[rows]))
            # argmax returns the first maximum: ties resolve to the lowest class, as in C++.
            predictions.append(logits.argmax(dim=1))
        return total, torch.cat(predictions)

    def loss(self, dataset):
        """Mean softmax cross-entropy, log-sum-exp(z) - z_label."""
        data = self._prepare(dataset)
        return self._evaluate(data)[0] / len(data)

    def predictions(self, dataset):
        return self._evaluate(self._prepare(dataset))[1]

    def accuracy(self, dataset):
        data = self._prepare(dataset)
        return int((self._evaluate(data)[1] == data.labels).sum()) / len(data)

    def confusion_matrix(self, dataset):
        """confusion[actual, predicted], counted with one bincount over flattened pairs."""
        data = self._prepare(dataset)
        k = self.output_size
        pairs = data.labels * k + self._evaluate(data)[1]
        return torch.bincount(pairs, minlength=k * k).reshape(k, k)

    def gradient(self, dataset):
        """Gradient of the mean loss, in the flat layout; chunk means are recombined by size."""
        data = self._prepare(dataset)
        flat = torch.zeros_like(self._flat)
        for start in range(0, len(data), CHUNK):
            rows = slice(start, start + CHUNK)
            labels = data.labels[rows]
            flat += self._gradient(data.features(rows), self._one_hot(labels)) * len(labels)
        return flat / len(data)

    def autograd_gradient(self, dataset):
        """An independent oracle: torch.autograd of the forward loss on a cloned leaf.

        It never calls the hand-derived backward pass and leaves the parameters and .grad
        untouched.
        """
        data = self._prepare(dataset)
        with torch.enable_grad():
            flat = self._flat.clone().requires_grad_(True)
            weights, biases = self._views(flat)
            total = sum(
                _cross_entropy_sum(
                    self._forward_cache(data.features(rows), weights, biases)[0][-1],
                    data.labels[rows],
                )
                for rows in (slice(s, s + CHUNK) for s in range(0, len(data), CHUNK))
            )
            (gradient,) = torch.autograd.grad(total / len(data), flat)
        return gradient.detach()

    @torch.no_grad()
    def fit(self, dataset, config=None):
        """Mini-batch gradient descent with the scalar model's local random.Random shuffle."""
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
                batch = order[begin : begin + batch_size]
                targets = self._one_hot(data.labels[batch])
                self._flat -= config.learning_rate * self._gradient(data.features(batch), targets)
            epoch_loss = self._evaluate(data)[0] / n
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return NetworkTrainingResult(epoch, True, tuple(history))
        return NetworkTrainingResult(config.max_epochs, False, tuple(history))
