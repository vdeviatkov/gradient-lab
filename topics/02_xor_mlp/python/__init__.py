"""Public API for xor multilayer perceptron; implementation lives in model.py."""

from .model import (
    MlpParameters,
    MlpTrainingResult,
    XorMlp,
)

__all__ = ["MlpParameters", "MlpTrainingResult", "XorMlp"]
