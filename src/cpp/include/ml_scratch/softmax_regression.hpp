#pragma once

#include "ml_scratch/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ml_scratch {

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

struct SoftmaxTrainingConfig {
    double learning_rate{0.1};
    std::size_t max_epochs{20};
    // Zero means one full-dataset batch per epoch.
    std::size_t batch_size{64};
    bool shuffle{true};
    std::uint32_t seed{0};
    // L2 penalty coefficient. Applied to the weights only, never to the biases.
    double l2_regularization{0.0};
    double target_loss{0.0};
};

struct SoftmaxTrainingResult {
    std::size_t epochs;
    bool converged;
    std::vector<double> loss_per_epoch;

    bool operator==(const SoftmaxTrainingResult&) const = default;
};

// Multiclass logistic regression: one linear score per class, turned into a distribution by
// softmax and fitted by cross-entropy.
//
// This is the same function a single linear layer of FeedForwardNetwork computes under
// softmax_cross_entropy, and the tests check that the two agree. It exists separately because it
// indexes a LabeledDataset of class indices in place rather than materializing one-hot targets and
// per-batch copies, which is what makes full MNIST practical.
//
// Parameters flatten to weights row by row — all weights of class 0, then class 1, and so on —
// followed by the biases, matching that layer's layout.
class SoftmaxRegression {
  public:
    SoftmaxRegression(std::size_t feature_count, std::size_t class_count);

    [[nodiscard]] std::vector<double> scores(const std::vector<double>& features) const;
    [[nodiscard]] std::vector<double>
    predict_probabilities(const std::vector<double>& features) const;
    // Index of the largest score. Ties resolve to the lowest class index.
    [[nodiscard]] std::size_t predict(const std::vector<double>& features) const;

    [[nodiscard]] double cross_entropy(const LabeledDataset& dataset,
                                       double l2_regularization = 0.0) const;
    [[nodiscard]] double accuracy(const LabeledDataset& dataset) const;
    [[nodiscard]] ConfusionCounts confusion_matrix(const LabeledDataset& dataset) const;
    [[nodiscard]] MulticlassMetrics evaluate(const LabeledDataset& dataset) const;

    // Gradient of the dataset loss with respect to every parameter, in the flat layout above.
    [[nodiscard]] std::vector<double> gradient(const LabeledDataset& dataset,
                                               double l2_regularization = 0.0) const;

    SoftmaxTrainingResult fit(const LabeledDataset& dataset,
                              const SoftmaxTrainingConfig& config = {});

    [[nodiscard]] std::vector<double> parameters() const;
    void set_parameters(const std::vector<double>& values);
    [[nodiscard]] const FeatureMatrix& weights() const noexcept { return weights_; }
    [[nodiscard]] const std::vector<double>& biases() const noexcept { return biases_; }
    [[nodiscard]] std::size_t feature_count() const noexcept { return feature_count_; }
    [[nodiscard]] std::size_t class_count() const noexcept { return biases_.size(); }
    [[nodiscard]] std::size_t parameter_count() const noexcept {
        return biases_.size() * (feature_count_ + 1);
    }

  private:
    void validate_dataset(const LabeledDataset& dataset) const;

    std::size_t feature_count_;
    // weights_[class][feature]
    FeatureMatrix weights_;
    std::vector<double> biases_;
};

} // namespace ml_scratch
