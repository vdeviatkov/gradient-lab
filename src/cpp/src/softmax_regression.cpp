#include "ml_scratch/softmax_regression.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace ml_scratch {
namespace {

double safe_ratio(const std::size_t numerator, const std::size_t denominator) {
    return denominator == 0 ? 0.0
                            : static_cast<double>(numerator) / static_cast<double>(denominator);
}

} // namespace

MulticlassMetrics multiclass_metrics(const ConfusionCounts& confusion) {
    if (confusion.empty()) {
        throw std::invalid_argument("confusion matrix must not be empty");
    }
    for (const auto& row : confusion) {
        if (row.size() != confusion.size()) {
            throw std::invalid_argument("confusion matrix must be square");
        }
    }

    const std::size_t classes = confusion.size();
    MulticlassMetrics metrics;
    metrics.per_class_precision.assign(classes, 0.0);
    metrics.per_class_recall.assign(classes, 0.0);
    metrics.per_class_f1.assign(classes, 0.0);

    std::size_t correct = 0;
    std::size_t total = 0;
    for (std::size_t actual = 0; actual < classes; ++actual) {
        correct += confusion[actual][actual];
        for (const std::size_t count : confusion[actual]) {
            total += count;
        }
    }
    if (total == 0) {
        throw std::invalid_argument("confusion matrix must contain at least one sample");
    }
    metrics.accuracy = safe_ratio(correct, total);

    for (std::size_t label = 0; label < classes; ++label) {
        std::size_t predicted_total = 0;
        std::size_t actual_total = 0;
        for (std::size_t other = 0; other < classes; ++other) {
            predicted_total += confusion[other][label];
            actual_total += confusion[label][other];
        }
        const double precision = safe_ratio(confusion[label][label], predicted_total);
        const double recall = safe_ratio(confusion[label][label], actual_total);
        metrics.per_class_precision[label] = precision;
        metrics.per_class_recall[label] = recall;
        metrics.per_class_f1[label] =
            (precision + recall) == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);

        metrics.macro_precision += precision;
        metrics.macro_recall += recall;
        metrics.macro_f1 += metrics.per_class_f1[label];
    }
    const auto class_scale = static_cast<double>(classes);
    metrics.macro_precision /= class_scale;
    metrics.macro_recall /= class_scale;
    metrics.macro_f1 /= class_scale;
    return metrics;
}

SoftmaxRegression::SoftmaxRegression(const std::size_t feature_count, const std::size_t class_count)
    : feature_count_(feature_count), weights_(class_count, std::vector<double>(feature_count, 0.0)),
      biases_(class_count, 0.0) {
    if (feature_count == 0) {
        throw std::invalid_argument("feature_count must be positive");
    }
    if (class_count < 2) {
        throw std::invalid_argument("class_count must be at least two");
    }
    // Zero initialization is valid here: the loss is convex in the parameters and the classes are
    // distinguished by their own score rows, so there is no symmetry to break.
}

void SoftmaxRegression::validate_dataset(const LabeledDataset& dataset) const {
    const DatasetShape shape = validate_labeled_dataset(dataset);
    if (shape.feature_count != feature_count_) {
        throw std::invalid_argument("sample feature count does not match the model");
    }
    if (shape.class_count > class_count()) {
        throw std::invalid_argument("dataset contains a label outside the class count");
    }
}

std::vector<double> SoftmaxRegression::scores(const std::vector<double>& features) const {
    if (features.size() != feature_count_) {
        throw std::invalid_argument("feature count does not match the model");
    }

    std::vector<double> logits(class_count(), 0.0);
    for (std::size_t label = 0; label < class_count(); ++label) {
        double total = biases_[label];
        const std::vector<double>& row = weights_[label];
        for (std::size_t feature = 0; feature < feature_count_; ++feature) {
            if (!std::isfinite(features[feature])) {
                throw std::invalid_argument("features must be finite");
            }
            total += row[feature] * features[feature];
        }
        logits[label] = total;
    }
    return logits;
}

std::vector<double>
SoftmaxRegression::predict_probabilities(const std::vector<double>& features) const {
    std::vector<double> logits = scores(features);
    // Subtracting the largest logit leaves the distribution unchanged and keeps exp from
    // overflowing.
    const double largest = *std::max_element(logits.begin(), logits.end());
    double total = 0.0;
    for (double& value : logits) {
        value = std::exp(value - largest);
        total += value;
    }
    for (double& value : logits) {
        value /= total;
    }
    return logits;
}

std::size_t SoftmaxRegression::predict(const std::vector<double>& features) const {
    const std::vector<double> logits = scores(features);
    std::size_t best = 0;
    for (std::size_t label = 1; label < logits.size(); ++label) {
        if (logits[label] > logits[best]) {
            best = label;
        }
    }
    return best;
}

double SoftmaxRegression::cross_entropy(const LabeledDataset& dataset,
                                        const double l2_regularization) const {
    validate_dataset(dataset);
    if (!std::isfinite(l2_regularization) || l2_regularization < 0.0) {
        throw std::invalid_argument("l2_regularization must be finite and non-negative");
    }

    double total = 0.0;
    for (const auto& sample : dataset) {
        const std::vector<double> logits = scores(sample.features);
        const double largest = *std::max_element(logits.begin(), logits.end());
        double sum_of_exponentials = 0.0;
        for (const double logit : logits) {
            sum_of_exponentials += std::exp(logit - largest);
        }
        // log-sum-exp: the shift makes this exact even when a logit is large.
        total += largest + std::log(sum_of_exponentials) - logits[sample.label];
    }
    double loss = total / static_cast<double>(dataset.size());

    if (l2_regularization > 0.0) {
        double squared_norm = 0.0;
        for (const std::vector<double>& row : weights_) {
            for (const double weight : row) {
                squared_norm += weight * weight;
            }
        }
        loss += 0.5 * l2_regularization * squared_norm;
    }
    return loss;
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

std::vector<double> SoftmaxRegression::gradient(const LabeledDataset& dataset,
                                                const double l2_regularization) const {
    validate_dataset(dataset);
    if (!std::isfinite(l2_regularization) || l2_regularization < 0.0) {
        throw std::invalid_argument("l2_regularization must be finite and non-negative");
    }

    std::vector<double> flat(parameter_count(), 0.0);
    const double scale = 1.0 / static_cast<double>(dataset.size());
    const std::size_t bias_offset = class_count() * feature_count_;

    for (const auto& sample : dataset) {
        const std::vector<double> probabilities = predict_probabilities(sample.features);
        for (std::size_t label = 0; label < class_count(); ++label) {
            // Composing softmax with cross-entropy collapses the output delta to p - y.
            const double delta =
                scale * (probabilities[label] - (label == sample.label ? 1.0 : 0.0));
            const std::size_t row_offset = label * feature_count_;
            for (std::size_t feature = 0; feature < feature_count_; ++feature) {
                flat[row_offset + feature] += delta * sample.features[feature];
            }
            flat[bias_offset + label] += delta;
        }
    }

    if (l2_regularization > 0.0) {
        // The penalty covers the weights only; a biased intercept is not a complexity cost.
        for (std::size_t label = 0; label < class_count(); ++label) {
            const std::size_t row_offset = label * feature_count_;
            for (std::size_t feature = 0; feature < feature_count_; ++feature) {
                flat[row_offset + feature] += l2_regularization * weights_[label][feature];
            }
        }
    }
    return flat;
}

SoftmaxTrainingResult SoftmaxRegression::fit(const LabeledDataset& dataset,
                                             const SoftmaxTrainingConfig& config) {
    validate_dataset(dataset);
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.max_epochs == 0) {
        throw std::invalid_argument("max_epochs must be positive");
    }
    if (!std::isfinite(config.l2_regularization) || config.l2_regularization < 0.0) {
        throw std::invalid_argument("l2_regularization must be finite and non-negative");
    }
    if (!std::isfinite(config.target_loss) || config.target_loss < 0.0) {
        throw std::invalid_argument("target_loss must be finite and non-negative");
    }

    std::vector<std::size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random_engine{config.seed};
    const std::size_t batch_size =
        config.batch_size == 0 ? dataset.size() : std::min(config.batch_size, dataset.size());

    std::vector<double> history;
    history.reserve(config.max_epochs);
    std::vector<double> probabilities(class_count(), 0.0);

    for (std::size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
        if (config.shuffle && batch_size < dataset.size()) {
            std::shuffle(order.begin(), order.end(), random_engine);
        }

        for (std::size_t begin = 0; begin < dataset.size(); begin += batch_size) {
            const std::size_t end = std::min(begin + batch_size, dataset.size());
            const double inverse_batch = 1.0 / static_cast<double>(end - begin);
            FeatureMatrix weight_gradients(class_count(), std::vector<double>(feature_count_, 0.0));
            std::vector<double> bias_gradients(class_count(), 0.0);

            // Samples are indexed in place; no batch is ever copied, which is what keeps a
            // 60000 x 784 dataset workable.
            for (std::size_t position = begin; position < end; ++position) {
                const LabeledSample& sample = dataset[order[position]];
                probabilities = predict_probabilities(sample.features);
                for (std::size_t label = 0; label < class_count(); ++label) {
                    const double delta = probabilities[label] - (label == sample.label ? 1.0 : 0.0);
                    if (delta == 0.0) {
                        continue;
                    }
                    std::vector<double>& row = weight_gradients[label];
                    for (std::size_t feature = 0; feature < feature_count_; ++feature) {
                        row[feature] += delta * sample.features[feature];
                    }
                    bias_gradients[label] += delta;
                }
            }

            for (std::size_t label = 0; label < class_count(); ++label) {
                std::vector<double>& row = weights_[label];
                const std::vector<double>& gradient_row = weight_gradients[label];
                for (std::size_t feature = 0; feature < feature_count_; ++feature) {
                    const double penalty = config.l2_regularization * row[feature];
                    row[feature] -=
                        config.learning_rate * (gradient_row[feature] * inverse_batch + penalty);
                }
                biases_[label] -= config.learning_rate * bias_gradients[label] * inverse_batch;
            }
        }

        const double epoch_loss = cross_entropy(dataset, config.l2_regularization);
        if (!std::isfinite(epoch_loss)) {
            throw std::runtime_error("training diverged to a non-finite loss");
        }
        history.push_back(epoch_loss);
        if (config.target_loss > 0.0 && epoch_loss <= config.target_loss) {
            return {epoch, true, std::move(history)};
        }
    }
    return {config.max_epochs, false, std::move(history)};
}

std::vector<double> SoftmaxRegression::parameters() const {
    std::vector<double> flat;
    flat.reserve(parameter_count());
    for (const std::vector<double>& row : weights_) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    flat.insert(flat.end(), biases_.begin(), biases_.end());
    return flat;
}

void SoftmaxRegression::set_parameters(const std::vector<double>& values) {
    if (values.size() != parameter_count()) {
        throw std::invalid_argument("parameter vector has the wrong size");
    }
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("parameters must be finite");
        }
    }

    std::size_t cursor = 0;
    for (std::vector<double>& row : weights_) {
        for (double& weight : row) {
            weight = values[cursor++];
        }
    }
    for (double& bias : biases_) {
        bias = values[cursor++];
    }
}

} // namespace ml_scratch
