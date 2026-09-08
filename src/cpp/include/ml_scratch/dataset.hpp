#pragma once

#include <cstddef>
#include <vector>

namespace ml_scratch {

// Row-major unlabeled design matrix: one inner vector per sample.
using FeatureMatrix = std::vector<std::vector<double>>;

// Throws std::invalid_argument unless the matrix is non-empty, rectangular, and finite.
// Returns the number of features per sample.
[[nodiscard]] std::size_t validate_feature_matrix(const FeatureMatrix& data);

[[nodiscard]] std::vector<double> column_means(const FeatureMatrix& data);

[[nodiscard]] double squared_euclidean_distance(const std::vector<double>& left,
                                                const std::vector<double>& right);

} // namespace ml_scratch
