"""Public API for perceptron and logic gates; implementation lives in model.py."""

from .model import (
    Perceptron,
    TrainingResult,
)

__all__ = ["Perceptron", "TrainingResult"]
