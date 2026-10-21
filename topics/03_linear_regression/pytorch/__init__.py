"""PyTorch linear regression and conditioning diagnostics."""

from .model import (
    LinearRegression,
    Optimizer,
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
    "LinearRegression",
    "Optimizer",
    "TrainingConfig",
    "TrainingResult",
    "condition_number",
    "hessian",
    "hessian_eigenvalue_range",
    "largest_hessian_eigenvalue",
    "solve_linear_system",
    "standardize",
]
