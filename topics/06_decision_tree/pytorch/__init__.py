"""PyTorch decision tree classification."""

from .model import (
    DecisionTreeClassifier,
    DecisionTreeConfig,
    entropy_impurity,
    gini_impurity,
)

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeConfig",
    "entropy_impurity",
    "gini_impurity",
]
