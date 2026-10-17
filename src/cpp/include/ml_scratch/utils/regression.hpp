#pragma once

#include "ml_scratch/linear_regression.hpp"

namespace ml_scratch {

// Independent least-squares reference: solves (X^T X) theta = X^T y by pivoted elimination.
// Singular designs are rejected. Writes parameters only after a successful solve.
void fit_closed_form(LinearRegression& model, const RegressionDataset& dataset);

// Improvement over predicting the target mean; undefined for constant targets.
[[nodiscard]] double r_squared(const LinearRegression& model, const RegressionDataset& dataset);

} // namespace ml_scratch
