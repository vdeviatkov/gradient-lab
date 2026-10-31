"""The topic 07 NumPy network with class-index labels over byte-stored MNIST images."""

import math
import random

from ml_scratch.numpy.backpropagation.model import FeedForwardNetwork
from ml_scratch.numpy.softmax_mnist.model import ByteImageDataset, DenseDataset
from ml_scratch.pure.backpropagation.model import Loss, NetworkTrainingConfig, NetworkTrainingResult

import numpy as np

# Rows per evaluation chunk: bounds the float64 copy of the images and every layer's activations.
CHUNK = 4096


def dense_examples(examples, feature_count):
    """Small (features, label) pairs, such as test data, as an already-scaled DenseDataset."""
    examples = list(examples)
    if not examples:
        raise ValueError("dataset must not be empty")
    x = np.asarray([features for features, _ in examples], dtype=np.float64)
    labels = np.asarray([label for _, label in examples])
    if x.ndim != 2 or x.shape[1] != feature_count or not np.isfinite(x).all():
        raise ValueError(f"expected {feature_count} finite features per sample")
    if labels.dtype.kind not in "iu" or (labels < 0).any():
        raise ValueError("labels must be non-negative integers")
    return DenseDataset(x, labels.astype(np.int64))


class ClassificationNetwork(FeedForwardNetwork):
    """FeedForwardNetwork with the C++ class-index overloads, over byte or dense datasets.

    The forward cache and the batched backward pass are the topic 07 network's: a batch's labels
    become one-hot rows, and `_gradient` does the rest. What this class adds is the dataset
    handling MNIST needs: images stay uint8 and are scaled one batch or chunk at a time.
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
        targets = np.zeros((len(labels), self.output_size))
        targets[np.arange(len(labels)), labels] = 1.0
        return targets

    def _chunks(self, data):
        for start in range(0, len(data), CHUNK):
            rows = slice(start, start + CHUNK)
            yield self._forward_cache(data.features(rows))[0][-1], data.labels[rows]

    def _evaluate(self, data):
        """Summed cross-entropy and the predicted classes, chunk by chunk."""
        total, predictions = 0.0, []
        for logits, labels in self._chunks(data):
            largest = logits.max(axis=1)
            log_normalizer = largest + np.log(np.exp(logits - largest[:, None]).sum(axis=1))
            total += float((log_normalizer - logits[np.arange(len(labels)), labels]).sum())
            # argmax returns the first maximum: ties resolve to the lowest class, as in C++.
            predictions.append(logits.argmax(axis=1))
        return total, np.concatenate(predictions)

    def loss(self, dataset):
        """Mean softmax cross-entropy, log-sum-exp(z) - z_label."""
        data = self._prepare(dataset)
        return self._evaluate(data)[0] / len(data)

    def predictions(self, dataset):
        return self._evaluate(self._prepare(dataset))[1]

    def accuracy(self, dataset):
        data = self._prepare(dataset)
        return float(np.mean(self._evaluate(data)[1] == data.labels))

    def confusion_matrix(self, dataset):
        """confusion[actual, predicted], counted with one bincount over flattened pairs."""
        data = self._prepare(dataset)
        k = self.output_size
        pairs = data.labels * k + self._evaluate(data)[1]
        return np.bincount(pairs, minlength=k * k).reshape(k, k)

    def gradient(self, dataset):
        """Gradient of the mean loss, in the flat layout; chunk means are recombined by size."""
        data = self._prepare(dataset)
        flat = np.zeros_like(self._flat)
        for start in range(0, len(data), CHUNK):
            rows = slice(start, start + CHUNK)
            labels = data.labels[rows]
            flat += self._gradient(data.features(rows), self._one_hot(labels)) * len(labels)
        return flat / len(data)

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
            # A diverging run overflows on its way to the non-finite check below, which reports it.
            with np.errstate(over="ignore", invalid="ignore"):
                for begin in range(0, n, batch_size):
                    batch = order[begin : begin + batch_size]
                    targets = self._one_hot(data.labels[batch])
                    self._flat -= config.learning_rate * self._gradient(
                        data.features(batch), targets
                    )
                epoch_loss = self._evaluate(data)[0] / n
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return NetworkTrainingResult(epoch, True, tuple(history))
        return NetworkTrainingResult(config.max_epochs, False, tuple(history))
