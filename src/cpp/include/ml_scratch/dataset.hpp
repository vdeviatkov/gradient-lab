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

// A sample with a zero-based class index. Multiclass models share this representation.
struct LabeledSample {
    std::vector<double> features;
    std::size_t label;

    bool operator==(const LabeledSample&) const = default;
};

using LabeledDataset = std::vector<LabeledSample>;

struct DatasetShape {
    std::size_t feature_count;
    // One past the largest label seen, so labels are always 0 .. class_count - 1.
    std::size_t class_count;

    bool operator==(const DatasetShape&) const = default;
};

// Throws std::invalid_argument unless the dataset is non-empty, rectangular, and finite.
[[nodiscard]] DatasetShape validate_labeled_dataset(const LabeledDataset& dataset);

[[nodiscard]] std::vector<std::size_t> class_counts(const LabeledDataset& dataset,
                                                    std::size_t class_count);

} // namespace ml_scratch
