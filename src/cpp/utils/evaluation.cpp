#include "ml_scratch/convolution.hpp"
#include "ml_scratch/kmeans.hpp"
#include "ml_scratch/logistic_regression.hpp"
#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/pca.hpp"
#include "ml_scratch/perceptron.hpp"
#include "ml_scratch/softmax_regression.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ml_scratch {

double BinaryPerceptron::accuracy(const LogicDataset& dataset) const noexcept {
    std::size_t correct = 0;
    for (const auto& [features, target] : dataset) {
        correct += predict(features) == target ? 1U : 0U;
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

ClassificationMetrics BinaryLogisticRegression::evaluate(const BinaryClassificationDataset& dataset,
                                                         const double threshold) const {
    validate_dataset(dataset);
    std::vector<int> targets;
    std::vector<int> predictions;
    targets.reserve(dataset.size());
    predictions.reserve(dataset.size());
    for (const auto& sample : dataset) {
        targets.push_back(sample.target);
        predictions.push_back(predict(sample.features, threshold));
    }
    return classification_metrics(targets, predictions);
}

double SoftmaxRegression::accuracy(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

ConfusionCounts SoftmaxRegression::confusion_matrix(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    ConfusionCounts confusion(class_count(), std::vector<std::size_t>(class_count(), 0));
    for (const auto& sample : dataset) {
        ++confusion[sample.label][predict(sample.features)];
    }
    return confusion;
}

MulticlassMetrics SoftmaxRegression::evaluate(const LabeledDataset& dataset) const {
    return multiclass_metrics(confusion_matrix(dataset));
}

double FeedForwardNetwork::accuracy(const SupervisedDataset& dataset) const {
    validate_for_network(dataset);
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        const auto expected = static_cast<std::size_t>(
            std::distance(sample.targets.begin(),
                          std::max_element(sample.targets.begin(), sample.targets.end())));
        if (predict_class(sample.features) == expected) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

double FeedForwardNetwork::accuracy(const LabeledDataset& dataset) const {
    static_cast<void>(validate_labeled(dataset));

    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict_class(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

std::vector<std::vector<std::size_t>>
FeedForwardNetwork::confusion_matrix(const LabeledDataset& dataset) const {
    static_cast<void>(validate_labeled(dataset));

    std::vector<std::vector<std::size_t>> confusion(output_size(),
                                                    std::vector<std::size_t>(output_size(), 0));
    for (const auto& sample : dataset) {
        ++confusion[sample.label][predict_class(sample.features)];
    }
    return confusion;
}

double ConvolutionalNetwork::accuracy(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict_class(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

std::vector<std::vector<std::size_t>>
ConvolutionalNetwork::confusion_matrix(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    const std::size_t classes = output_shape().size();
    std::vector<std::vector<std::size_t>> confusion(classes, std::vector<std::size_t>(classes, 0));
    for (const auto& sample : dataset) {
        ++confusion[sample.label][predict_class(sample.features)];
    }
    return confusion;
}

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
