"""Text checkpoints in the C++ FeedForwardNetwork format, so each language loads the other's files.

The format is whitespace-separated tokens:

    ml_scratch_feedforward 3
    loss <code>
    layers <count>
    <input> <output> <activation code> <normalization code> <dropout rate> <residual flag>
    ...
    parameters <count>
    <one value per line, flat layout>
    statistics <count>
    <one value per line>

Codes are the C++ enum positions, which the Python enums list in the same order. Values carry 17
significant digits, which round-trips every IEEE-754 double exactly, so a restored network is the
same function and not an approximation of it. The writer formats values as C++'s
`std::setprecision(17)` does, so a Python checkpoint is byte-for-byte what C++ writes for the same
network. Normalization, dropout, residual skips, and the leaky rectifier belong to later C++
topics; these networks do not implement them, so the reader rejects a checkpoint that uses them
rather than load a different function.
"""

import math
from pathlib import Path

from .model import Activation, DenseLayer, Loss

MAGIC = "ml_scratch_feedforward"
VERSION = 3
ACTIVATIONS = tuple(Activation)  # identity, sigmoid, tanh, relu: the C++ enum order
LOSSES = tuple(Loss)  # mean squared error, binary and softmax cross-entropy
_CPP_ACTIVATION_COUNT = 5  # C++ also defines a leaky rectifier, code 4.


def save_checkpoint(path, layers, loss, parameters):
    """Write a version-3 checkpoint; `parameters` is any iterable of floats in the flat layout."""
    values = [float(value) for value in parameters]
    if not all(map(math.isfinite, values)):
        raise ValueError("parameters must be finite")
    lines = [f"{MAGIC} {VERSION}", f"loss {LOSSES.index(loss)}", f"layers {len(layers)}"]
    for layer in layers:
        # Normalization none, dropout rate 0, no residual skip.
        code = ACTIVATIONS.index(layer.activation)
        lines.append(f"{layer.input_size} {layer.output_size} {code} 0 0 0")
    lines.append(f"parameters {len(values)}")
    # %.17g is exactly what a C++ stream prints at precision 17.
    lines.extend(format(value, ".17g") for value in values)
    lines.append("statistics 0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def load_checkpoint(path):
    """Read a version 1-3 checkpoint; returns (layers, loss, parameters as a list of floats).

    Raises ValueError for a malformed or unsupported file; the caller's network constructor then
    re-validates the architecture and its parameter count.
    """
    tokens = iter(Path(path).read_text(encoding="ascii").split())

    def malformed(reason):
        return ValueError(f"malformed checkpoint {path}: {reason}")

    def expect(keyword):
        if next(tokens, None) != keyword:
            raise malformed(f"expected {keyword}")

    def size(maximum=None):
        token = next(tokens, None)
        if token is None or not token.isdigit():
            raise malformed("expected a size")
        value = int(token)
        if maximum is not None and value > maximum:
            raise malformed(f"code {value} is out of range")
        return value

    def number():
        token = next(tokens, None)
        try:
            return float(token)
        except (TypeError, ValueError):
            raise malformed("truncated or non-numeric value") from None

    expect(MAGIC)
    version = size()
    if not 1 <= version <= VERSION:
        raise ValueError(f"unsupported checkpoint version {version} in {path}")
    expect("loss")
    loss = LOSSES[size(len(LOSSES) - 1)]
    expect("layers")
    layers = []
    for _ in range(size()):
        input_size, output_size = size(), size()
        activation = size(_CPP_ACTIVATION_COUNT - 1)
        # Version 1 predates normalization and dropout and version 2 the skip; absent fields
        # take their defaults, exactly as the C++ reader does.
        normalization, dropout = (size(), number()) if version >= 2 else (0, 0.0)
        residual = size() if version >= 3 else 0
        if activation >= len(ACTIVATIONS) or normalization or dropout or residual:
            raise ValueError(
                f"checkpoint {path} uses a layer feature from a later topic (leaky ReLU, "
                "normalization, dropout, or a residual skip) that this network does not implement"
            )
        try:
            layers.append(DenseLayer(input_size, output_size, ACTIVATIONS[activation]))
        except ValueError as error:
            raise malformed(str(error)) from None
    if not layers:
        raise malformed("no layers")
    expect("parameters")
    expected = sum(layer.output_size * (layer.input_size + 1) for layer in layers)
    count = size()
    if count != expected:
        raise malformed("parameter count does not match the architecture")
    parameters = [number() for _ in range(count)]
    if version >= 2:
        expect("statistics")
        # Running statistics exist only for batch normalization, which is rejected above.
        if size() != 0:
            raise malformed("statistics count does not match the architecture")
    return tuple(layers), loss, parameters
