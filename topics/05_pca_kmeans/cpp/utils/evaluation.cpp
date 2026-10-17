#include "ml_scratch/kmeans.hpp"
#include "ml_scratch/pca.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ml_scratch {

double PrincipalComponentAnalysis::reconstruction_error(const FeatureMatrix& data) const {
    require_fitted();
    const std::size_t features = validate_feature_matrix(data);
    if (features != feature_count()) {
        throw std::invalid_argument("feature count does not match the fitted model");
    }

    double squared_error = 0.0;
    for (const auto& sample : data) {
        squared_error += squared_euclidean_distance(sample, inverse_transform(transform(sample)));
    }
    return squared_error / static_cast<double>(data.size() * features);
}

double PrincipalComponentAnalysis::cumulative_explained_variance_ratio() const noexcept {
    return std::accumulate(explained_variance_ratio_.begin(), explained_variance_ratio_.end(), 0.0);
}

double cluster_purity(const std::vector<std::size_t>& assignments,
                      const std::vector<std::size_t>& true_labels) {
    if (assignments.empty()) {
        throw std::invalid_argument("assignments must not be empty");
    }
    if (assignments.size() != true_labels.size()) {
        throw std::invalid_argument("assignments and true_labels must have equal sizes");
    }

    const std::size_t cluster_count = *std::max_element(assignments.begin(), assignments.end()) + 1;
    const std::size_t label_count = *std::max_element(true_labels.begin(), true_labels.end()) + 1;
    std::vector<std::vector<std::size_t>> counts(cluster_count,
                                                 std::vector<std::size_t>(label_count, 0));
    for (std::size_t index = 0; index < assignments.size(); ++index) {
        ++counts[assignments[index]][true_labels[index]];
    }

    std::size_t majority_total = 0;
    for (const auto& cluster_counts : counts) {
        majority_total += *std::max_element(cluster_counts.begin(), cluster_counts.end());
    }
    return static_cast<double>(majority_total) / static_cast<double>(assignments.size());
}

} // namespace ml_scratch
