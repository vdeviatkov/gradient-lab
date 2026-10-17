#include "ml_scratch/utils/classification_metrics.hpp"
#include <cmath>
#include <stdexcept>

namespace ml_scratch {

namespace {

double safe_ratio(const std::size_t numerator, const std::size_t denominator) {
    return denominator == 0 ? 0.0
                            : static_cast<double>(numerator) / static_cast<double>(denominator);
}

void require_binary_label(const int label, const char* message) {
    if (label != 0 && label != 1)
        throw std::invalid_argument(message);
}
void require_threshold(const double threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0)
        throw std::invalid_argument("threshold must be finite and between zero and one");
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

} // namespace ml_scratch
