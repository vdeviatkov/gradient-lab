"""Softmax regression whose training step is two fused Triton kernels on a CUDA device."""

import math
import os
import random
from dataclasses import dataclass

import torch
from ml_scratch.pure.softmax_mnist.model import SoftmaxTrainingConfig, SoftmaxTrainingResult

import triton

from .kernels import reduce_and_update, softmax_rows

BLOCK_M = 16  # Rows per forward program.
BLOCK_R = 16  # Batch rows per step of the gradient reduction.
BLOCK_D = 32  # Features per tile. Ten classes pad to BLOCK_K = 16.
# Rows per evaluation launch: bounds the (rows, classes) delta buffer for a full split.
CHUNK = 8192
_INTEGER_TYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def interpreting():
    """TRITON_INTERPRET=1 runs the same kernels on CPU tensors through Triton's interpreter."""
    return os.environ.get("TRITON_INTERPRET") == "1"


def default_device():
    if interpreting():
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("the Triton backend needs a CUDA device or TRITON_INTERPRET=1")
    return torch.device("cuda")


@dataclass(frozen=True)
class DeviceDataset:
    """Features resident on the device: uint8 pixels (scaled in-kernel) or float64 rows."""

    x: torch.Tensor  # (n, D) uint8 or float64, contiguous
    labels: torch.Tensor  # (n,) int64

    def __post_init__(self):
        if self.x.ndim != 2 or not len(self.x) or self.x.dtype not in (torch.uint8, torch.float64):
            raise ValueError("features must be a nonempty uint8 or float64 matrix")
        if self.labels.shape != (len(self.x),) or self.labels.dtype != torch.int64:
            raise ValueError("labels must be one int64 per row")
        if not self.x.is_contiguous() or self.x.device != self.labels.device:
            raise ValueError("features must be contiguous and share the labels' device")

    @property
    def bytes(self):
        return self.x.dtype == torch.uint8

    def __len__(self):
        return len(self.labels)


def byte_images(images, labels, feature_count, device=None):
    """Move MNIST byte strings to the device once; they stay uint8 there."""
    device = device or default_device()
    pixels = torch.frombuffer(bytearray(b"".join(images)), dtype=torch.uint8)
    return DeviceDataset(
        pixels.reshape(len(images), feature_count).to(device),
        torch.tensor(labels, dtype=torch.int64, device=device),
    )


def _non_negative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


class SoftmaxRegression:
    """The scalar model's API with parameters in one flat float64 device tensor."""

    def __init__(self, feature_count, class_count, device=None):
        for value, name, low in (
            (feature_count, "feature_count", 1),
            (class_count, "class_count", 2),
        ):
            if type(value) is not int or value < low:
                raise ValueError(f"{name} must be an integer >= {low}")
        if class_count > 128:
            raise ValueError("this kernel keeps all classes in one tile; use at most 128")
        self.feature_count = feature_count
        self.device = device or default_device()
        self._flat = torch.zeros(
            class_count * (feature_count + 1), dtype=torch.float64, device=self.device
        )
        cut = class_count * feature_count
        self.weights = self._flat[:cut].view(class_count, feature_count)
        self.biases = self._flat[cut:]
        self._block_k = triton.next_power_of_2(max(class_count, 16))

    @property
    def class_count(self):
        return len(self.biases)

    @property
    def parameter_count(self):
        return len(self._flat)

    def parameters(self):
        return self._flat.cpu()

    def set_parameters(self, values):
        values = torch.as_tensor(values, dtype=torch.float64).detach().reshape(-1)
        if values.shape != self._flat.shape:
            raise ValueError("parameter vector has the wrong size")
        if not torch.isfinite(values).all():
            raise ValueError("parameters must be finite")
        self._flat.copy_(values)

    def _prepare(self, dataset):
        if not isinstance(dataset, DeviceDataset):
            examples = list(dataset)
            if not examples:
                raise ValueError("dataset must not be empty")
            try:
                x = torch.tensor([list(f) for f, _ in examples], dtype=torch.float64)
            except (TypeError, ValueError) as error:
                raise ValueError("features must be rectangular numbers") from error
            y = torch.as_tensor([label for _, label in examples])
            if not torch.isfinite(x).all():
                raise ValueError("features must be finite")
            if y.dtype not in _INTEGER_TYPES or (y < 0).any():
                raise ValueError("labels must be non-negative integers")
            dataset = DeviceDataset(x.to(self.device), y.to(torch.int64).to(self.device))
        if dataset.x.shape[1] != self.feature_count:
            raise ValueError("sample feature count does not match the model")
        if dataset.x.device.type != self.device.type:
            raise ValueError("dataset and model must be on the same device")
        if int(dataset.labels.max()) >= self.class_count:
            raise ValueError("dataset contains a label outside the class count")
        return dataset

    def _forward(self, data, rows):
        """Launch softmax_rows for the given row indices; returns (delta, loss, predictions)."""
        count = len(rows)
        delta = torch.empty((count, self.class_count), dtype=torch.float64, device=self.device)
        loss = torch.empty(count, dtype=torch.float64, device=self.device)
        predictions = torch.empty(count, dtype=torch.int64, device=self.device)
        softmax_rows[(triton.cdiv(count, BLOCK_M),)](
            data.x, rows, data.labels, self.weights, self.biases, delta, loss, predictions,
            count, self.class_count, D=self.feature_count,
            BYTES=data.bytes, BLOCK_M=BLOCK_M, BLOCK_K=self._block_k, BLOCK_D=BLOCK_D,
        )  # fmt: skip
        return delta, loss, predictions

    def _reduce(self, data, rows, delta, learning_rate, l2, apply):
        """Launch reduce_and_update over one class tile and every feature tile."""
        if apply:
            grad_w = grad_b = self._flat  # Unused by the APPLY variant.
        else:
            grad_w = torch.empty_like(self.weights)
            grad_b = torch.empty_like(self.biases)
        grid = (
            triton.cdiv(self.class_count, self._block_k),
            triton.cdiv(self.feature_count, BLOCK_D),
        )
        # A device array, because Triton would pass Python floats to the kernel as fp32.
        hyper = torch.tensor([learning_rate, l2], dtype=torch.float64, device=self.device)
        reduce_and_update[grid](
            data.x, rows, delta, self.weights, self.biases, grad_w, grad_b,
            len(rows), self.class_count, hyper, D=self.feature_count,
            MAX_B=max(triton.next_power_of_2(len(rows)), BLOCK_R),
            BYTES=data.bytes, APPLY=apply, BLOCK_R=BLOCK_R, BLOCK_K=self._block_k, BLOCK_D=BLOCK_D,
        )  # fmt: skip
        return grad_w, grad_b

    def _all_rows(self, data):
        everything = torch.arange(len(data), dtype=torch.int64, device=self.device)
        for start in range(0, len(data), CHUNK):
            yield everything[start : start + CHUNK]

    def _row(self, features):
        x = torch.as_tensor(features, dtype=torch.float64).detach().reshape(1, -1)
        if x.shape[1] != self.feature_count or not torch.isfinite(x).all():
            raise ValueError(f"expected {self.feature_count} finite features")
        zero = torch.zeros(1, dtype=torch.int64, device=self.device)
        return DeviceDataset(x.to(self.device), zero), zero

    def scores(self, features):
        # A single vector is not worth a kernel; this host-side product is for inspection only.
        return (
            torch.as_tensor(features, dtype=torch.float64) @ self.weights.cpu().T
            + self.biases.cpu()
        )

    def predict(self, features):
        data, rows = self._row(features)
        return int(self._forward(data, rows)[2][0])

    def predict_probabilities(self, features):
        data, rows = self._row(features)
        delta = self._forward(data, rows)[0][0].cpu()
        delta[0] += 1.0  # The placeholder label 0 was subtracted; add it back to recover p.
        return delta

    def cross_entropy(self, dataset, l2_regularization=0.0):
        """Mean per-row loss from the kernel, plus (l2 / 2) ||W||^2."""
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        total = sum(float(self._forward(data, rows)[1].sum()) for rows in self._all_rows(data))
        penalty = 0.5 * l2_regularization * float((self.weights * self.weights).sum())
        return total / len(data) + penalty

    def predictions(self, dataset):
        data = self._prepare(dataset)
        return torch.cat([self._forward(data, rows)[2] for rows in self._all_rows(data)]).cpu()

    def accuracy(self, dataset):
        data = self._prepare(dataset)
        return int((self.predictions(data) == data.labels.cpu()).sum()) / len(data)

    def confusion_matrix(self, dataset):
        data = self._prepare(dataset)
        k = self.class_count
        pairs = data.labels.cpu() * k + self.predictions(data)
        return torch.bincount(pairs, minlength=k * k).reshape(k, k)

    def gradient(self, dataset, l2_regularization=0.0):
        """Gradient of the mean objective from the same kernels, with the update switched off."""
        _non_negative(l2_regularization, "l2_regularization")
        data = self._prepare(dataset)
        rows = torch.arange(len(data), dtype=torch.int64, device=self.device)
        delta = self._forward(data, rows)[0]
        grad_w, grad_b = self._reduce(data, rows, delta, 0.0, l2_regularization, apply=False)
        return torch.cat((grad_w.reshape(-1), grad_b)).cpu()

    def fit(self, dataset, config=None):
        """Mini-batch SGD; the scalar shuffle stream's order is uploaded once per epoch."""
        config = config or SoftmaxTrainingConfig()
        data = self._prepare(dataset)
        n = len(data)
        batch_size = n if config.batch_size == 0 else min(config.batch_size, n)
        order = list(range(n))
        shuffler = random.Random(config.seed)
        history = []
        for epoch in range(1, config.max_epochs + 1):
            if config.shuffle and batch_size < n:
                shuffler.shuffle(order)
            device_order = torch.tensor(order, dtype=torch.int64, device=self.device)
            for begin in range(0, n, batch_size):
                rows = device_order[begin : begin + batch_size]
                delta = self._forward(data, rows)[0]
                self._reduce(
                    data, rows, delta, config.learning_rate, config.l2_regularization, apply=True
                )
            epoch_loss = self.cross_entropy(data, config.l2_regularization)
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return SoftmaxTrainingResult(epoch, True, tuple(history))
        return SoftmaxTrainingResult(config.max_epochs, False, tuple(history))
