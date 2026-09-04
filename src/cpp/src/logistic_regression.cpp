#include "ml_scratch/logistic_regression.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace ml_scratch {
namespace {

void require_binary_label(const int label, const char* message) {
    if (label != 0 && label != 1) {
        throw std::invalid_argument(message);
    }
}

void require_threshold(const double threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("threshold must be finite and between zero and one");
    }
}

double safe_ratio(const std::size_t numerator, const std::size_t denominator) {
    return denominator == 0 ? 0.0
                            : static_cast<double>(numerator) / static_cast<double>(denominator);
}

double sigmoid(const double score) {
    if (score >= 0.0) {
        return 1.0 / (1.0 + std::exp(-score));
    }
    const double exponential = std::exp(score);
    return exponential / (1.0 + exponential);
}

double softplus(const double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

} // namespace

ClassificationMetrics classification_metrics(const std::vector<int>& targets,
                                             const std::vector<int>& predictions) {
    if (targets.empty()) {
        throw std::invalid_argument("targets must not be empty");
    }
    if (targets.size() != predictions.size()) {
        throw std::invalid_argument("targets and predictions must have equal sizes");
    }

    ConfusionMatrix confusion;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        require_binary_label(targets[index], "targets must contain only zero or one");
        require_binary_label(predictions[index], "predictions must contain only zero or one");
        if (targets[index] == 1 && predictions[index] == 1) {
            ++confusion.true_positives;
        } else if (targets[index] == 0 && predictions[index] == 0) {
            ++confusion.true_negatives;
        } else if (targets[index] == 0) {
            ++confusion.false_positives;
        } else {
            ++confusion.false_negatives;
        }
    }

    const std::size_t correct = confusion.true_positives + confusion.true_negatives;
    const double accuracy = safe_ratio(correct, targets.size());
    const double precision =
        safe_ratio(confusion.true_positives, confusion.true_positives + confusion.false_positives);
    const double recall =
        safe_ratio(confusion.true_positives, confusion.true_positives + confusion.false_negatives);
    const double f1 =
        (precision + recall) == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
    return {confusion, accuracy, precision, recall, f1};
}

ThresholdSelectionResult select_threshold_for_f1(const std::vector<int>& targets,
                                                 const std::vector<double>& probabilities,
                                                 const std::vector<double>& candidate_thresholds) {
    if (targets.empty()) {
        throw std::invalid_argument("targets must not be empty");
    }
    if (targets.size() != probabilities.size()) {
        throw std::invalid_argument("targets and probabilities must have equal sizes");
    }
    if (candidate_thresholds.empty()) {
        throw std::invalid_argument("candidate_thresholds must not be empty");
    }
    for (const int target : targets) {
        require_binary_label(target, "targets must contain only zero or one");
    }
    for (const double probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
            throw std::invalid_argument("probabilities must be finite and between zero and one");
        }
    }

    bool has_best = false;
    ThresholdSelectionResult best{};
    for (const double threshold : candidate_thresholds) {
        require_threshold(threshold);
        std::vector<int> predictions;
        predictions.reserve(probabilities.size());
        for (const double probability : probabilities) {
            predictions.push_back(probability >= threshold ? 1 : 0);
        }
        const ClassificationMetrics metrics = classification_metrics(targets, predictions);

        const bool better_f1 = !has_best || metrics.f1 > best.metrics.f1;
        const bool tied_f1 = has_best && metrics.f1 == best.metrics.f1;
        const double distance = std::abs(threshold - 0.5);
        const double best_distance = std::abs(best.threshold - 0.5);
        const bool better_tie =
            tied_f1 &&
            (distance < best_distance || (distance == best_distance && threshold < best.threshold));
        if (better_f1 || better_tie) {
            best = {threshold, metrics};
            has_best = true;
        }
    }
    return best;
}

BinaryLogisticRegression::BinaryLogisticRegression(const std::size_t feature_count)
    : weights_(feature_count, 0.0) {
    if (feature_count == 0) {
        throw std::invalid_argument("feature_count must be positive");
    }
}

void BinaryLogisticRegression::validate_dataset(const BinaryClassificationDataset& dataset) const {
    if (dataset.empty()) {
        throw std::invalid_argument("dataset must not be empty");
    }
    for (const auto& sample : dataset) {
        if (sample.features.size() != feature_count()) {
            throw std::invalid_argument("sample feature count does not match the model");
        }
        require_binary_label(sample.target, "targets must contain only zero or one");
        for (const double feature : sample.features) {
            if (!std::isfinite(feature)) {
                throw std::invalid_argument("features must be finite");
            }
        }
    }
}

double BinaryLogisticRegression::decision_score(const std::vector<double>& features) const {
    if (features.size() != feature_count()) {
        throw std::invalid_argument("feature count does not match the model");
    }

    double score = bias_;
    for (std::size_t feature = 0; feature < feature_count(); ++feature) {
        if (!std::isfinite(features[feature])) {
            throw std::invalid_argument("features must be finite");
        }
        score += weights_[feature] * features[feature];
    }
    return score;
}

double BinaryLogisticRegression::predict_probability(const std::vector<double>& features) const {
    return sigmoid(decision_score(features));
}

int BinaryLogisticRegression::predict(const std::vector<double>& features,
                                      const double threshold) const {
    require_threshold(threshold);
    return predict_probability(features) >= threshold ? 1 : 0;
}

double BinaryLogisticRegression::binary_cross_entropy(const BinaryClassificationDataset& dataset,
                                                      const double positive_class_weight) const {
    validate_dataset(dataset);
    if (!std::isfinite(positive_class_weight) || positive_class_weight <= 0.0) {
        throw std::invalid_argument("positive_class_weight must be finite and positive");
    }

    double loss_sum = 0.0;
    for (const auto& sample : dataset) {
        const double score = decision_score(sample.features);
        loss_sum += sample.target == 1 ? positive_class_weight * softplus(-score) : softplus(score);
    }
    return loss_sum / static_cast<double>(dataset.size());
}

ClassificationMetrics BinaryLogisticRegression::evaluate(const BinaryClassificationDataset& dataset,
                                                         const double threshold) const {
    validate_dataset(dataset);
    require_threshold(threshold);
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

LogisticTrainingResult BinaryLogisticRegression::fit(const BinaryClassificationDataset& dataset,
                                                     const LogisticTrainingConfig& config) {
    validate_dataset(dataset);
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.max_epochs == 0) {
        throw std::invalid_argument("max_epochs must be positive");
    }
    if (!std::isfinite(config.positive_class_weight) || config.positive_class_weight <= 0.0) {
        throw std::invalid_argument("positive_class_weight must be finite and positive");
    }
    if (!std::isfinite(config.target_loss) || config.target_loss < 0.0) {
        throw std::invalid_argument("target_loss must be finite and non-negative");
    }

    std::fill(weights_.begin(), weights_.end(), 0.0);
    bias_ = 0.0;
    std::vector<std::size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random_engine{config.seed};
    std::vector<double> history;
    history.reserve(config.max_epochs);
    const std::size_t batch_size =
        config.batch_size == 0 ? dataset.size() : std::min(config.batch_size, dataset.size());

    for (std::size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
        if (config.shuffle && batch_size < dataset.size()) {
            std::shuffle(order.begin(), order.end(), random_engine);
        }

        for (std::size_t begin = 0; begin < dataset.size(); begin += batch_size) {
            const std::size_t end = std::min(begin + batch_size, dataset.size());
            const double inverse_batch_size = 1.0 / static_cast<double>(end - begin);
            std::vector<double> weight_gradients(feature_count(), 0.0);
            double bias_gradient = 0.0;

            for (std::size_t position = begin; position < end; ++position) {
                const auto& sample = dataset[order[position]];
                const double probability = predict_probability(sample.features);
                const double score_gradient =
                    sample.target == 1 ? config.positive_class_weight * (probability - 1.0)
                                       : probability;
                bias_gradient += score_gradient;
                for (std::size_t feature = 0; feature < feature_count(); ++feature) {
                    weight_gradients[feature] += score_gradient * sample.features[feature];
                }
            }

            bias_ -= config.learning_rate * bias_gradient * inverse_batch_size;
            for (std::size_t feature = 0; feature < feature_count(); ++feature) {
                weights_[feature] -=
                    config.learning_rate * weight_gradients[feature] * inverse_batch_size;
            }
        }

        const double loss = binary_cross_entropy(dataset, config.positive_class_weight);
        if (!std::isfinite(loss)) {
            throw std::runtime_error("training diverged to a non-finite loss");
        }
        history.push_back(loss);
        if (config.target_loss > 0.0 && loss <= config.target_loss) {
            return {epoch, true, std::move(history)};
        }
    }
    return {config.max_epochs, false, std::move(history)};
}

} // namespace ml_scratch
