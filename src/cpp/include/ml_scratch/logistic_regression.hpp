#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ml_scratch {

struct BinaryClassificationSample {
    std::vector<double> features;
    int target;

    bool operator==(const BinaryClassificationSample&) const = default;
};

using BinaryClassificationDataset = std::vector<BinaryClassificationSample>;

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

struct LogisticTrainingConfig {
    double learning_rate{0.1};
    std::size_t max_epochs{1'000};
    // Zero means one full-dataset batch. A positive value enables mini-batches.
    std::size_t batch_size{0};
    bool shuffle{true};
    std::uint32_t seed{0};
    double positive_class_weight{1.0};
    double target_loss{0.0};
};

struct LogisticTrainingResult {
    std::size_t epochs;
    bool converged;
    std::vector<double> loss_per_epoch;

    bool operator==(const LogisticTrainingResult&) const = default;
};

[[nodiscard]] ClassificationMetrics classification_metrics(const std::vector<int>& targets,
                                                           const std::vector<int>& predictions);

[[nodiscard]] ThresholdSelectionResult
select_threshold_for_f1(const std::vector<int>& targets, const std::vector<double>& probabilities,
                        const std::vector<double>& candidate_thresholds);

class BinaryLogisticRegression {
  public:
    explicit BinaryLogisticRegression(std::size_t feature_count);

    [[nodiscard]] double decision_score(const std::vector<double>& features) const;
    [[nodiscard]] double predict_probability(const std::vector<double>& features) const;
    [[nodiscard]] int predict(const std::vector<double>& features, double threshold = 0.5) const;
    [[nodiscard]] double binary_cross_entropy(const BinaryClassificationDataset& dataset,
                                              double positive_class_weight = 1.0) const;
    [[nodiscard]] ClassificationMetrics evaluate(const BinaryClassificationDataset& dataset,
                                                 double threshold = 0.5) const;

    LogisticTrainingResult fit(const BinaryClassificationDataset& dataset,
                               const LogisticTrainingConfig& config = {});

    [[nodiscard]] const std::vector<double>& weights() const noexcept { return weights_; }
    [[nodiscard]] double bias() const noexcept { return bias_; }
    [[nodiscard]] std::size_t feature_count() const noexcept { return weights_.size(); }

  private:
    void validate_dataset(const BinaryClassificationDataset& dataset) const;

    std::vector<double> weights_;
    double bias_{0.0};
};

} // namespace ml_scratch
