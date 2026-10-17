#pragma once

#include <cstddef>
#include <vector>

namespace ml_scratch {

struct ConfusionMatrix {
    std::size_t true_negatives{};
    std::size_t false_positives{};
    std::size_t false_negatives{};
    std::size_t true_positives{};

    bool operator==(const ConfusionMatrix&) const = default;
};

struct ClassificationMetrics {
    ConfusionMatrix confusion;
    double accuracy{};
    double precision{};
    double recall{};
    double f1{};

    bool operator==(const ClassificationMetrics&) const = default;
};

struct ThresholdSelectionResult {
    double threshold;
    ClassificationMetrics metrics;

    bool operator==(const ThresholdSelectionResult&) const = default;
};

[[nodiscard]] ClassificationMetrics classification_metrics(const std::vector<int>& targets,
                                                           const std::vector<int>& predictions);

[[nodiscard]] ThresholdSelectionResult
select_threshold_for_f1(const std::vector<int>& targets, const std::vector<double>& probabilities,
                        const std::vector<double>& candidate_thresholds);

using ConfusionCounts = std::vector<std::vector<std::size_t>>;

struct MulticlassMetrics {
    double accuracy{};
    // Unweighted means over classes, so a rare class counts as much as a common one.
    double macro_precision{};
    double macro_recall{};
    double macro_f1{};
    std::vector<double> per_class_precision;
    std::vector<double> per_class_recall;
    std::vector<double> per_class_f1;

    bool operator==(const MulticlassMetrics&) const = default;
};

// confusion[actual][predicted]. A ratio with a zero denominator is defined as zero.
[[nodiscard]] MulticlassMetrics multiclass_metrics(const ConfusionCounts& confusion);

} // namespace ml_scratch
