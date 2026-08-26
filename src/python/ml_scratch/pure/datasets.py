"""Small immutable datasets used by the introductory experiments."""

from typing import Literal

Features = tuple[float, ...]
Example = tuple[Features, int]
Dataset = tuple[Example, ...]
LogicGate = Literal["and", "or", "xor"]

_INPUTS: tuple[Features, ...] = (
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
)

_TARGETS: dict[LogicGate, tuple[int, ...]] = {
    "and": (0, 0, 0, 1),
    "or": (0, 1, 1, 1),
    "xor": (0, 1, 1, 0),
}


def logic_gate_dataset(gate: LogicGate) -> Dataset:
    """Return the complete two-input truth table for a named logic gate."""
    try:
        targets = _TARGETS[gate]
    except KeyError as error:
        choices = ", ".join(_TARGETS)
        raise ValueError(f"unknown gate {gate!r}; expected one of: {choices}") from error
    return tuple(zip(_INPUTS, targets, strict=True))
