"""Accuracy, per-class and macro precision, recall, and F1 from a confusion matrix."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MulticlassMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]


def _ratio(numerator, denominator):
    # A zero denominator is defined as zero, so a class that is never predicted shows up
    # explicitly instead of as an undefined value.
    return numerator / denominator if denominator else 0.0


def _f1(precision, recall):
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def multiclass_metrics(confusion):
    """Metrics from confusion[actual][predicted]; macro averages weight every class equally."""
    rows = [list(row) for row in confusion]
    k = len(rows)
    if not k or any(len(row) != k for row in rows):
        raise ValueError("confusion matrix must be square and nonempty")
    if any(type(count) is not int or count < 0 for row in rows for count in row):
        raise ValueError("confusion counts must be non-negative integers")
    total = sum(map(sum, rows))
    if not total:
        raise ValueError("confusion matrix must contain at least one sample")
    precision = tuple(_ratio(rows[c][c], sum(row[c] for row in rows)) for c in range(k))
    recall = tuple(_ratio(rows[c][c], sum(rows[c])) for c in range(k))
    f1 = tuple(_f1(p, r) for p, r in zip(precision, recall, strict=True))
    return MulticlassMetrics(
        sum(rows[c][c] for c in range(k)) / total,
        sum(precision) / k,
        sum(recall) / k,
        sum(f1) / k,
        precision,
        recall,
        f1,
    )
