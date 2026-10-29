"""From-scratch softmax regression on sparse rows, with multiclass evaluation metrics."""

from .metrics import MulticlassMetrics, multiclass_metrics
from .model import (
    SoftmaxRegression,
    SoftmaxTrainingConfig,
    SoftmaxTrainingResult,
    SparseDataset,
    softmax,
    sparse_dataset,
)

__all__ = [
    "MulticlassMetrics",
    "SoftmaxRegression",
    "SoftmaxTrainingConfig",
    "SoftmaxTrainingResult",
    "SparseDataset",
    "multiclass_metrics",
    "softmax",
    "sparse_dataset",
]
