"""Public API for binary logistic regression; implementation lives in model.py."""

from .model import (
    BinaryLogisticRegression,
    ClassificationMetrics,
    ConfusionMatrix,
    LogisticTrainingConfig,
    LogisticTrainingResult,
    ThresholdSelectionResult,
    classification_metrics,
    select_threshold_for_f1,
)

__all__ = [
    "BinaryLogisticRegression",
    "ClassificationMetrics",
    "ConfusionMatrix",
    "LogisticTrainingConfig",
    "LogisticTrainingResult",
    "ThresholdSelectionResult",
    "classification_metrics",
    "select_threshold_for_f1",
]
