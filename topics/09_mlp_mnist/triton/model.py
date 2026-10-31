"""The topic 07 network on a CUDA device: every layer's forward and backward pass is a kernel."""

import math
import random

import torch
from ml_scratch.pure.backpropagation.checkpoint import load_checkpoint, save_checkpoint
from ml_scratch.pure.backpropagation.model import (
    Activation,
    Loss,
    NetworkTrainingConfig,
    NetworkTrainingResult,
    initial_parameters,
    validate_architecture,
)
from ml_scratch.triton.softmax_mnist.kernels import reduce_and_update, softmax_rows
from ml_scratch.triton.softmax_mnist.model import DeviceDataset, default_device

import triton

from .kernels import dense_rows, hidden_delta

BLOCK_M = 16  # Rows per forward and delta program.
BLOCK_R = 16  # Batch rows per step of the gradient reduction.
BLOCK_D = 32  # Inputs per tile.
BLOCK_K = 16  # Units per tile, in every kernel but softmax_rows, which holds all classes.
# Rows per evaluation pass: bounds every layer's (rows, units) activation buffer.
CHUNK = 8192
_ACTIVATION_CODES = {activation: code for code, activation in enumerate(Activation)}
_INTEGER_TYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class ClassificationNetwork:
    """The topic 07 architecture, initialization, flat layout, and checkpoints, trained by kernels.

    Parameters live in one flat float64 device tensor with per-layer weight and bias views, in
    the scalar layout, so `parameters()` and checkpoints are interchangeable with every other
    backend. Datasets are topic 08's DeviceDataset: uint8 pixels stay uint8 on the device.
    """

    def __init__(self, layers, loss, seed=0, device=None):
        self.layers = validate_architecture(layers, loss)
        if loss is not Loss.SOFTMAX_CROSS_ENTROPY:
            raise ValueError("the class-index methods require the softmax cross-entropy loss")
        if self.layers[-1].output_size > 128:
            raise ValueError("softmax_rows keeps all classes in one tile; use at most 128")
        self.loss_function = loss
        self.device = device or default_device()
        self._flat = torch.tensor(
            initial_parameters(self.layers, seed), dtype=torch.float64, device=self.device
        )
        self.weights, self.biases, cursor = [], [], 0
        for layer in self.layers:
            size = layer.output_size * layer.input_size
            self.weights.append(
                self._flat[cursor : cursor + size].view(layer.output_size, layer.input_size)
            )
            self.biases.append(self._flat[cursor + size : cursor + size + layer.output_size])
            cursor += size + layer.output_size
        self._class_block = triton.next_power_of_2(max(self.output_size, 16))

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
        return self._flat.cpu()

    def set_parameters(self, values):
        values = torch.as_tensor(values, dtype=torch.float64).detach().reshape(-1)
        if values.shape != self._flat.shape:
            raise ValueError("parameter vector has the wrong size")
        if not torch.isfinite(values).all():
            raise ValueError("parameters must be finite")
        self._flat.copy_(values)

    def save(self, path):
        """Write a text checkpoint in the C++ format, shared with every other backend."""
        save_checkpoint(path, self.layers, self.loss_function, self._flat.tolist())

    @classmethod
    def load(cls, path, device=None):
        layers, loss, values = load_checkpoint(path)
        network = cls(layers, loss, device=device)
        network.set_parameters(values)
        return network

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
        if dataset.x.shape[1] != self.input_size:
            raise ValueError("sample feature count does not match the network input size")
        if dataset.x.device.type != self.device.type:
            raise ValueError("dataset and network must be on the same device")
        if int(dataset.labels.max()) >= self.output_size:
            raise ValueError("dataset contains a label outside the network output size")
        return dataset

    def _forward(self, data, rows):
        """Every hidden activation, then the output delta, loss, and predictions for `rows`.

        Layer 0 reads the dataset through `rows`; every later layer reads the previous layer's
        activation buffer in order, so its row indices are simply 0..B-1.
        """
        count = len(rows)
        sequence = torch.arange(count, dtype=torch.int64, device=self.device)
        inputs = [(data.x, rows, data.bytes)]
        for layer, weights, biases in zip(self.layers[:-1], self.weights, self.biases):
            out = torch.empty((count, layer.output_size), dtype=torch.float64, device=self.device)
            source, source_rows, source_bytes = inputs[-1]
            grid = (triton.cdiv(count, BLOCK_M), triton.cdiv(layer.output_size, BLOCK_K))
            dense_rows[grid](
                source, source_rows, weights, biases, out, count, layer.output_size,
                D=layer.input_size, BYTES=source_bytes,
                ACTIVATION=_ACTIVATION_CODES[layer.activation],
                BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            )  # fmt: skip
            inputs.append((out, sequence, False))
        source, source_rows, source_bytes = inputs[-1]
        # softmax_rows reads labels through the same indices as its inputs, so a hidden-layer
        # input needs the batch's labels gathered into 0..B-1 order.
        labels = data.labels if len(inputs) == 1 else data.labels[rows]
        delta = torch.empty((count, self.output_size), dtype=torch.float64, device=self.device)
        loss = torch.empty(count, dtype=torch.float64, device=self.device)
        predictions = torch.empty(count, dtype=torch.int64, device=self.device)
        softmax_rows[(triton.cdiv(count, BLOCK_M),)](
            source, source_rows, labels, self.weights[-1], self.biases[-1],
            delta, loss, predictions, count, self.output_size, D=self.layers[-1].input_size,
            BYTES=source_bytes, BLOCK_M=BLOCK_M, BLOCK_K=self._class_block, BLOCK_D=BLOCK_D,
        )  # fmt: skip
        return inputs, delta, loss, predictions

    def _backward(self, inputs, delta, learning_rate, apply):
        """Every layer's delta from the top down, then each layer's reduction.

        All deltas are computed before any update, because delta_prev needs the weights the
        forward pass used. Returns the per-layer gradients when not applying them.
        """
        count = len(delta)
        deltas = [delta]
        for index in range(len(self.layers) - 1, 0, -1):
            below = self.layers[index - 1]
            out = torch.empty((count, below.output_size), dtype=torch.float64, device=self.device)
            grid = (triton.cdiv(count, BLOCK_M), triton.cdiv(below.output_size, BLOCK_K))
            hidden_delta[grid](
                deltas[0], self.weights[index], inputs[index][0], out, count, below.output_size,
                K=self.layers[index].output_size, ACTIVATION=_ACTIVATION_CODES[below.activation],
                BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_K, BLOCK_K=BLOCK_K,
            )  # fmt: skip
            deltas.insert(0, out)
        # A device array, because Triton would pass Python floats to the kernel as fp32.
        hyper = torch.tensor([learning_rate, 0.0], dtype=torch.float64, device=self.device)
        gradients = []
        for layer, weights, biases, (source, source_rows, source_bytes), layer_delta in zip(
            self.layers, self.weights, self.biases, inputs, deltas, strict=True
        ):
            if apply:
                grad_w = grad_b = self._flat  # Unused by the APPLY variant.
            else:
                grad_w, grad_b = torch.empty_like(weights), torch.empty_like(biases)
            grid = (triton.cdiv(layer.output_size, BLOCK_K), triton.cdiv(layer.input_size, BLOCK_D))
            reduce_and_update[grid](
                source, source_rows, layer_delta, weights, biases, grad_w, grad_b,
                count, layer.output_size, hyper, D=layer.input_size,
                MAX_B=max(triton.next_power_of_2(count), BLOCK_R), BYTES=source_bytes,
                APPLY=apply, BLOCK_R=BLOCK_R, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            )  # fmt: skip
            gradients += [grad_w.reshape(-1), grad_b]
        return gradients

    def _all_rows(self, data):
        everything = torch.arange(len(data), dtype=torch.int64, device=self.device)
        for start in range(0, len(data), CHUNK):
            yield everything[start : start + CHUNK]

    def _evaluate(self, data):
        """Summed cross-entropy and the predicted classes, chunk by chunk."""
        total, predictions = 0.0, []
        for rows in self._all_rows(data):
            _, _, loss, predicted = self._forward(data, rows)
            total += float(loss.sum())
            predictions.append(predicted)
        return total, torch.cat(predictions).cpu()

    def loss(self, dataset):
        """Mean softmax cross-entropy, log-sum-exp(z) - z_label."""
        data = self._prepare(dataset)
        return self._evaluate(data)[0] / len(data)

    def predictions(self, dataset):
        return self._evaluate(self._prepare(dataset))[1]

    def accuracy(self, dataset):
        data = self._prepare(dataset)
        return int((self._evaluate(data)[1] == data.labels.cpu()).sum()) / len(data)

    def confusion_matrix(self, dataset):
        """confusion[actual, predicted]."""
        data = self._prepare(dataset)
        k = self.output_size
        pairs = data.labels.cpu() * k + self._evaluate(data)[1]
        return torch.bincount(pairs, minlength=k * k).reshape(k, k)

    def gradient(self, dataset):
        """Gradient of the mean loss from the training kernels, with the update switched off."""
        data = self._prepare(dataset)
        rows = torch.arange(len(data), dtype=torch.int64, device=self.device)
        inputs, delta, _, _ = self._forward(data, rows)
        return torch.cat(self._backward(inputs, delta, 0.0, apply=False)).cpu()

    def fit(self, dataset, config=None):
        """Mini-batch gradient descent; the scalar shuffle stream's order is uploaded per epoch."""
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
            device_order = torch.tensor(order, dtype=torch.int64, device=self.device)
            for begin in range(0, n, batch_size):
                rows = device_order[begin : begin + batch_size]
                inputs, delta, _, _ = self._forward(data, rows)
                self._backward(inputs, delta, config.learning_rate, apply=True)
            epoch_loss = self._evaluate(data)[0] / n
            if not math.isfinite(epoch_loss):
                raise ArithmeticError("training diverged to a non-finite loss")
            history.append(epoch_loss)
            if config.target_loss > 0 and epoch_loss <= config.target_loss:
                return NetworkTrainingResult(epoch, True, tuple(history))
        return NetworkTrainingResult(config.max_epochs, False, tuple(history))
