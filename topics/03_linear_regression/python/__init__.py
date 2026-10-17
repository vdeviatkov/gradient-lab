"""Public API for linear regression; implementation lives in model.py."""

from .model import (
    Features,
    LinearRegression,
    Optimizer,
    RegressionDataset,
    RegressionSample,
    TrainingConfig,
    TrainingResult,
    condition_number,
    hessian,
    hessian_eigenvalue_range,
    largest_hessian_eigenvalue,
    solve_linear_system,
    standardize,
)

__all__ = [
    "Features",
    "LinearRegression",
    "Optimizer",
    "RegressionDataset",
    "RegressionSample",
    "TrainingConfig",
    "TrainingResult",
    "condition_number",
    "hessian",
    "hessian_eigenvalue_range",
    "largest_hessian_eigenvalue",
    "solve_linear_system",
    "standardize",
]
