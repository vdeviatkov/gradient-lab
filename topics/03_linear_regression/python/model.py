"""Multivariate linear regression: a closed-form reference and three gradient methods."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

Features = tuple[float, ...]
RegressionSample = tuple[Features, float]
RegressionDataset = tuple[RegressionSample, ...]


class Optimizer(Enum):
    """Which gradient method ``LinearRegression.fit`` runs.

    They differ only in how many examples one parameter update is computed from: all of them,
    one, or a configurable few. That single choice decides how many updates an epoch makes, how
    noisy each one is, and therefore what learning rate is usable.
    """

    BATCH = "batch"
    STOCHASTIC = "stochastic"
    MINI_BATCH = "mini-batch"


@dataclass(frozen=True)
class TrainingConfig:
    """Everything a gradient run needs, so a run is reproducible from this object alone."""

    optimizer: Optimizer = Optimizer.BATCH
    learning_rate: float = 0.01
    max_epochs: int = 1_000
    batch_size: int = 4
    shuffle: bool = True
    seed: int = 0
    target_mse: float = 0.0


@dataclass(frozen=True)
class TrainingResult:
    """Describe a completed run without hiding non-convergence or divergence."""

    epochs: int
    converged: bool
    mse_per_epoch: tuple[float, ...]
    diverged: bool = False


def solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Solve ``A z = b`` by Gaussian elimination with partial pivoting.

    Partial pivoting — swapping in the row with the largest leading magnitude at each step — is
    what keeps the elimination from dividing by a tiny number it did not have to. Without it this
    routine is numerically useless on anything but a well-behaved matrix.
    """
    size = len(vector)
    if any(len(row) != size for row in matrix) or len(matrix) != size:
        raise ValueError("the matrix must be square and match the vector's length")

    augmented = [list(row) + [value] for row, value in zip(matrix, vector, strict=True)]
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if math.isclose(augmented[best][pivot], 0.0, abs_tol=1e-12):
            raise ValueError("the system is singular to working precision")
        augmented[pivot], augmented[best] = augmented[best], augmented[pivot]

        leading = augmented[pivot][pivot]
        for row in range(pivot + 1, size):
            factor = augmented[row][pivot] / leading
            if factor == 0.0:
                continue
            for column in range(pivot, size + 1):
                augmented[row][column] -= factor * augmented[pivot][column]

    solution = [0.0] * size
    for row in reversed(range(size)):
        total = augmented[row][size]
        for column in range(row + 1, size):
            total -= augmented[row][column] * solution[column]
        solution[row] = total / augmented[row][row]
    return solution


def hessian(dataset: RegressionDataset) -> list[list[float]]:
    """Return the mean-squared-error Hessian, ``H = (2/n) X~^T X~``.

    The loss is a quadratic in the parameters, so its second derivative does not depend on where
    it is evaluated: one constant matrix describes the whole surface, and its eigenvalues decide
    both the largest usable step and how many steps are needed.
    """
    rows = _augmented_rows(dataset)
    size = len(rows[0])
    scale = 2.0 / len(rows)
    return [
        [scale * sum(row[first] * row[second] for row in rows) for second in range(size)]
        for first in range(size)
    ]


def _power_iteration(
    matrix: Sequence[Sequence[float]], *, iterations: int, tolerance: float
) -> float:
    """Return the largest eigenvalue magnitude of a symmetric matrix, by repeated multiplication.

    Each multiplication amplifies the component along the dominant eigenvector more than any
    other, so renormalizing and repeating converges to it, and the norm converges to the
    eigenvalue. Four lines, and enough for what is needed here.
    """
    size = len(matrix)
    vector = [1.0] * size
    eigenvalue = 0.0
    for _ in range(iterations):
        product = [
            sum(entry * value for entry, value in zip(row, vector, strict=True)) for row in matrix
        ]
        norm = math.sqrt(sum(value * value for value in product))
        if norm == 0.0:
            return 0.0
        vector = [value / norm for value in product]
        if abs(norm - eigenvalue) <= tolerance * max(norm, 1.0):
            return norm
        eigenvalue = norm
    return eigenvalue


def hessian_eigenvalue_range(
    dataset: RegressionDataset, *, iterations: int = 10_000, tolerance: float = 1e-15
) -> tuple[float, float]:
    """Return ``(smallest, largest)`` eigenvalue of the loss Hessian.

    The largest comes from power iteration directly. The smallest comes from applying the same
    routine to ``lambda_max * I - H``, whose dominant eigenvalue is ``lambda_max - lambda_min``;
    the Hessian is positive semi-definite, so this shift makes the wanted end the dominant one.

    Both ends matter and for different reasons. Full-batch gradient descent on a quadratic
    converges exactly when ``0 < learning_rate < 2 / lambda_max``, so the largest bounds the step.
    The ratio of the two is the condition number, and it bounds how many steps are needed: the
    error contracts by at most ``(kappa - 1) / (kappa + 1)`` per step at the best fixed rate, so
    the step count grows with ``kappa``.
    """
    matrix = hessian(dataset)
    largest = _power_iteration(matrix, iterations=iterations, tolerance=tolerance)
    size = len(matrix)
    shifted = [
        [(largest if first == second else 0.0) - matrix[first][second] for second in range(size)]
        for first in range(size)
    ]
    gap = _power_iteration(shifted, iterations=iterations, tolerance=tolerance)
    return max(largest - gap, 0.0), largest


def largest_hessian_eigenvalue(
    dataset: RegressionDataset, *, iterations: int = 10_000, tolerance: float = 1e-15
) -> float:
    """Return the largest eigenvalue of the loss Hessian, which bounds the usable step size."""
    return _power_iteration(hessian(dataset), iterations=iterations, tolerance=tolerance)


def condition_number(dataset: RegressionDataset) -> float:
    """Return the loss Hessian's condition number, or infinity when it is singular."""
    smallest, largest = hessian_eigenvalue_range(dataset)
    return largest / smallest if smallest > 0.0 else math.inf


def standardize(dataset: RegressionDataset) -> RegressionDataset:
    """Rescale every feature to zero mean and unit variance, leaving the targets alone.

    Gradient descent's step is shared by every coordinate, so a feature measured in thousands and
    one measured in units cannot both take a well-sized step. Standardizing removes the
    difference; the experiment measures what it is worth.
    """
    rows = [features for features, _ in dataset]
    count = len(rows)
    width = len(rows[0])
    means = [sum(row[index] for row in rows) / count for index in range(width)]
    deviations = []
    for index in range(width):
        variance = (
            sum((row[index] - means[index]) * (row[index] - means[index]) for row in rows) / count
        )
        deviations.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return tuple(
        (
            tuple(
                (value - mean) / deviation
                for value, mean, deviation in zip(features, means, deviations, strict=True)
            ),
            target,
        )
        for features, target in dataset
    )


def _augmented_rows(dataset: RegressionDataset) -> list[list[float]]:
    """Return the design matrix with a leading one per row, so the bias is just another weight."""
    if not dataset:
        raise ValueError("dataset must not be empty")
    width = len(dataset[0][0])
    if width == 0:
        raise ValueError("samples must have at least one feature")
    rows = []
    for features, _ in dataset:
        if len(features) != width:
            raise ValueError("every sample must have the same number of features")
        rows.append([1.0, *features])
    return rows


@dataclass
class LinearRegression:
    """``y = w . x + b``, fitted either exactly or by gradient descent."""

    feature_count: int
    weights: list[float] = field(init=False)
    bias: float = field(init=False)

    def __post_init__(self) -> None:
        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")
        # Zero initialization, not random: the loss is convex with a single minimum, so where the
        # search starts changes only the path, and starting at zero makes a run depend on nothing
        # but its data and its step size.
        self.weights = [0.0] * self.feature_count
        self.bias = 0.0

    # -- prediction ------------------------------------------------------------------------

    def predict(self, features: Sequence[float]) -> float:
        """Return the model's prediction for one feature vector."""
        self._validate_features(features)
        return (
            sum(weight * value for weight, value in zip(self.weights, features, strict=True))
            + self.bias
        )

    def predict_many(self, examples: Iterable[Features]) -> tuple[float, ...]:
        """Predict several feature vectors in their given order."""
        return tuple(self.predict(features) for features in examples)

    # -- metrics ---------------------------------------------------------------------------

    def mean_squared_error(self, dataset: RegressionDataset) -> float:
        """Return the mean squared error, the quantity every method here minimizes.

        The squares are written ``r * r`` rather than ``r ** 2`` on purpose. CPython's float
        exponentiation raises ``OverflowError`` when the result leaves the double range, while
        multiplication returns infinity — and a diverging run produces exactly such a residual.
        Infinity is the useful answer here, since it says the run diverged; an exception would
        have to be caught by every caller. A C++ implementation gets this for free.
        """
        examples = self._validated(dataset)
        total = 0.0
        for features, target in examples:
            residual = self.predict(features) - target
            total += residual * residual
        return total / len(examples)

    def r_squared(self, dataset: RegressionDataset) -> float:
        """Return the fraction of the target's variance the model explains.

        One means the predictions are exact; zero means the model does no better than always
        predicting the mean. It can be negative, which says the model is worse than that.
        """
        examples = self._validated(dataset)
        mean = sum(target for _, target in examples) / len(examples)
        total = sum((target - mean) * (target - mean) for _, target in examples)
        if total == 0.0:
            raise ValueError("r_squared is undefined when every target is identical")
        residual = sum(
            (self.predict(features) - target) * (self.predict(features) - target)
            for features, target in examples
        )
        return 1.0 - residual / total

    # -- gradient --------------------------------------------------------------------------

    def gradient(self, dataset: RegressionDataset) -> tuple[list[float], float]:
        """Return ``(d/dw, d/db)`` of the mean squared error, derived from the chain rule.

        With ``r = prediction - target``, the loss is ``mean(r**2)``, so ``dL/dr = 2r/n`` and the
        rest is the derivative of a linear map: ``dL/dw_k = mean(2 r x_k)`` and ``dL/db =
        mean(2 r)``.

        The factor of two is kept rather than absorbed, so that this really is the gradient of
        :meth:`mean_squared_error` and a central-difference test can say so. The C++ milestone
        takes the other common convention and descends on half the mean squared error, which
        cancels it; the two therefore mean different things by the same learning rate, and the
        experiment measures the factor of two that results.
        """
        examples = self._validated(dataset)
        scale = 2.0 / len(examples)
        weight_gradient = [0.0] * self.feature_count
        bias_gradient = 0.0
        for features, target in examples:
            residual = self.predict(features) - target
            bias_gradient += scale * residual
            for index, value in enumerate(features):
                weight_gradient[index] += scale * residual * value
        return weight_gradient, bias_gradient

    # -- fitting ---------------------------------------------------------------------------

    def fit_closed_form(self, dataset: RegressionDataset) -> None:
        """Solve the normal equations ``(X~^T X~) theta = X~^T y`` exactly.

        The minimum of a convex quadratic is where its gradient vanishes, and that condition is a
        linear system. Solving it gives the answer in one step with no learning rate and no
        epochs — which is why it is the reference every gradient method here is measured against.

        Raises ``ValueError`` when the normal equations are singular to working precision, which
        happens when two features are collinear: the data then does not determine the individual
        coefficients at all, and refusing is more useful than returning one arbitrary member of
        the solution set. Near-collinearity is the harder case, because nothing raises — the fit
        stays good while the individual coefficients become meaningless, and the experiment
        measures that.
        """
        self._validated(dataset)
        rows = _augmented_rows(dataset)
        size = self.feature_count + 1
        gram = [[0.0] * size for _ in range(size)]
        products = [0.0] * size
        for row, (_, target) in zip(rows, dataset, strict=True):
            for first in range(size):
                products[first] += row[first] * target
                for second in range(size):
                    gram[first][second] += row[first] * row[second]

        parameters = solve_linear_system(gram, products)
        self.bias = parameters[0]
        self.weights = list(parameters[1:])

    def fit(
        self, dataset: RegressionDataset, config: TrainingConfig | None = None
    ) -> TrainingResult:
        """Fit by gradient descent, reporting divergence rather than returning nonsense."""
        configuration = config or TrainingConfig()
        examples = self._validated(dataset)
        if configuration.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if configuration.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if configuration.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if configuration.target_mse < 0:
            raise ValueError("target_mse must be non-negative")

        if configuration.optimizer is Optimizer.BATCH:
            batch_size = len(examples)
        elif configuration.optimizer is Optimizer.STOCHASTIC:
            batch_size = 1
        else:
            batch_size = min(configuration.batch_size, len(examples))

        generator = random.Random(configuration.seed)
        order = list(range(len(examples)))
        history: list[float] = []

        for epoch in range(1, configuration.max_epochs + 1):
            if configuration.shuffle and batch_size < len(examples):
                generator.shuffle(order)

            for start in range(0, len(examples), batch_size):
                batch = tuple(examples[index] for index in order[start : start + batch_size])
                weight_gradient, bias_gradient = self.gradient(batch)
                for index in range(self.feature_count):
                    self.weights[index] -= configuration.learning_rate * weight_gradient[index]
                self.bias -= configuration.learning_rate * bias_gradient

            error = self.mean_squared_error(dataset)
            history.append(error)
            # A step too large for this problem makes the error grow without bound; saying so is
            # more useful than returning parameters that are infinite or not a number.
            if not math.isfinite(error):
                return TrainingResult(epoch, False, tuple(history), diverged=True)
            if error <= configuration.target_mse:
                return TrainingResult(epoch, True, tuple(history))

        return TrainingResult(configuration.max_epochs, False, tuple(history))

    # -- validation ------------------------------------------------------------------------

    def _validate_features(self, features: Sequence[float]) -> None:
        if len(features) != self.feature_count:
            raise ValueError(f"expected {self.feature_count} features, received {len(features)}")

    def _validated(self, dataset: RegressionDataset) -> tuple[RegressionSample, ...]:
        examples = tuple(dataset)
        if not examples:
            raise ValueError("dataset must not be empty")
        for features, target in examples:
            self._validate_features(features)
            if not math.isfinite(target):
                raise ValueError("targets must be finite")
        return examples
