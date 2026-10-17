"""Dependency-free pure-Python implementations."""

from ml_scratch.pure.linear_regression import LinearRegression, Optimizer
from ml_scratch.pure.linear_regression import TrainingConfig as RegressionTrainingConfig
from ml_scratch.pure.linear_regression import TrainingResult as RegressionTrainingResult
from ml_scratch.pure.mlp import MlpParameters, MlpTrainingResult, XorMlp
from ml_scratch.pure.perceptron import Perceptron, TrainingResult

__all__ = [
    "LinearRegression",
    "MlpParameters",
    "MlpTrainingResult",
    "Optimizer",
    "Perceptron",
    "RegressionTrainingConfig",
    "RegressionTrainingResult",
    "TrainingResult",
    "XorMlp",
]
