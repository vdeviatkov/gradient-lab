#include "ml_scratch/dataset.hpp"

#include <cmath>
#include <stdexcept>

namespace ml_scratch {

std::size_t validate_feature_matrix(const FeatureMatrix& data) {
    if (data.empty()) {
        throw std::invalid_argument("dataset must not be empty");
    }
    const std::size_t feature_count = data.front().size();
    if (feature_count == 0) {
        throw std::invalid_argument("samples must have at least one feature");
    }
    for (const auto& sample : data) {
        if (sample.size() != feature_count) {
            throw std::invalid_argument("all samples must have the same feature count");
        }
        for (const double feature : sample) {
            if (!std::isfinite(feature)) {
                throw std::invalid_argument("features must be finite");
            }
        }
    }
    return feature_count;
}

std::vector<double> column_means(const FeatureMatrix& data) {
    const std::size_t feature_count = validate_feature_matrix(data);
    std::vector<double> means(feature_count, 0.0);
    for (const auto& sample : data) {
        for (std::size_t feature = 0; feature < feature_count; ++feature) {
            means[feature] += sample[feature];
        }
    }
    for (double& mean : means) {
        mean /= static_cast<double>(data.size());
    }
    return means;
}

double squared_euclidean_distance(const std::vector<double>& left,
                                  const std::vector<double>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("vectors must have equal sizes");
    }

    double total = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const double difference = left[index] - right[index];
        total += difference * difference;
    }
    return total;
}

} // namespace ml_scratch
