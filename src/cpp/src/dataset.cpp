#include "ml_scratch/dataset.hpp"

#include <algorithm>
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

DatasetShape validate_labeled_dataset(const LabeledDataset& dataset) {
    if (dataset.empty()) {
        throw std::invalid_argument("dataset must not be empty");
    }

    DatasetShape shape{dataset.front().features.size(), 0};
    if (shape.feature_count == 0) {
        throw std::invalid_argument("samples must have at least one feature");
    }
    for (const auto& sample : dataset) {
        if (sample.features.size() != shape.feature_count) {
            throw std::invalid_argument("all samples must have the same feature count");
        }
        for (const double feature : sample.features) {
            if (!std::isfinite(feature)) {
                throw std::invalid_argument("features must be finite");
            }
        }
        shape.class_count = std::max(shape.class_count, sample.label + 1);
    }
    return shape;
}

std::vector<std::size_t> class_counts(const LabeledDataset& dataset,
                                      const std::size_t class_count) {
    const DatasetShape shape = validate_labeled_dataset(dataset);
    if (shape.class_count > class_count) {
        throw std::invalid_argument("dataset contains a label outside the class count");
    }

    std::vector<std::size_t> counts(class_count, 0);
    for (const auto& sample : dataset) {
        ++counts[sample.label];
    }
    return counts;
}

SupervisedShape validate_supervised_dataset(const SupervisedDataset& dataset) {
    if (dataset.empty()) {
        throw std::invalid_argument("dataset must not be empty");
    }

    const SupervisedShape shape{dataset.front().features.size(), dataset.front().targets.size()};
    if (shape.feature_count == 0) {
        throw std::invalid_argument("samples must have at least one feature");
    }
    if (shape.target_count == 0) {
        throw std::invalid_argument("samples must have at least one target");
    }
    for (const auto& sample : dataset) {
        if (sample.features.size() != shape.feature_count) {
            throw std::invalid_argument("all samples must have the same feature count");
        }
        if (sample.targets.size() != shape.target_count) {
            throw std::invalid_argument("all samples must have the same target count");
        }
        for (const double feature : sample.features) {
            if (!std::isfinite(feature)) {
                throw std::invalid_argument("features must be finite");
            }
        }
        for (const double target : sample.targets) {
            if (!std::isfinite(target)) {
                throw std::invalid_argument("targets must be finite");
            }
        }
    }
    return shape;
}

SupervisedDataset to_one_hot(const LabeledDataset& dataset, const std::size_t class_count) {
    const DatasetShape shape = validate_labeled_dataset(dataset);
    if (shape.class_count > class_count) {
        throw std::invalid_argument("dataset contains a label outside the class count");
    }
    if (class_count == 0) {
        throw std::invalid_argument("class_count must be positive");
    }

    SupervisedDataset encoded;
    encoded.reserve(dataset.size());
    for (const auto& sample : dataset) {
        std::vector<double> targets(class_count, 0.0);
        targets[sample.label] = 1.0;
        encoded.push_back({sample.features, std::move(targets)});
    }
    return encoded;
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
