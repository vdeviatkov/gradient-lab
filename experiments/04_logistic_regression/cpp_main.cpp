#include "ml_scratch/logistic_regression.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260904;

const ml_scratch::BinaryClassificationDataset training_data{
    {{-2.8}, 0}, {{-2.4}, 0}, {{-2.0}, 0}, {{-1.6}, 0}, {{-1.2}, 0}, {{-0.8}, 0},
    {{-0.4}, 0}, {{0.0}, 0},  {{0.3}, 0},  {{0.6}, 0},  {{0.9}, 0},  {{1.2}, 0},
    {{0.4}, 1},  {{1.0}, 1},  {{1.6}, 1},  {{2.2}, 1},
};

const ml_scratch::BinaryClassificationDataset validation_data{
    {{-1.5}, 0}, {{-0.5}, 0}, {{0.2}, 0}, {{0.7}, 0}, {{1.1}, 0},
    {{1.5}, 0},  {{0.5}, 1},  {{0.9}, 1}, {{1.4}, 1},
};

const ml_scratch::BinaryClassificationDataset test_data{
    {{-2.2}, 0}, {{-1.4}, 0}, {{-0.7}, 0}, {{-0.1}, 0}, {{0.4}, 0},
    {{0.8}, 0},  {{1.2}, 0},  {{1.7}, 0},  {{0.6}, 1},  {{1.5}, 1},
};

void print_metrics(const std::string_view name, const ml_scratch::ClassificationMetrics& metrics) {
    const auto& confusion = metrics.confusion;
    std::cout << name << ": accuracy=" << metrics.accuracy << ", precision=" << metrics.precision
              << ", recall=" << metrics.recall << ", F1=" << metrics.f1
              << ", confusion=[TN=" << confusion.true_negatives
              << ", FP=" << confusion.false_positives << ", FN=" << confusion.false_negatives
              << ", TP=" << confusion.true_positives << "]\n";
}

} // namespace

int main() {
    ml_scratch::LogisticTrainingConfig config;
    config.learning_rate = 0.1;
    config.max_epochs = 2'000;
    config.batch_size = 4;
    config.seed = seed;
    config.target_loss = 0.30;

    ml_scratch::BinaryLogisticRegression model{1};
    const auto result = model.fit(training_data, config);

    std::vector<int> validation_targets;
    std::vector<double> validation_probabilities;
    for (const auto& sample : validation_data) {
        validation_targets.push_back(sample.target);
        validation_probabilities.push_back(model.predict_probability(sample.features));
    }
    const std::vector<double> candidate_thresholds{0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                                                   0.50, 0.55, 0.60, 0.65, 0.70, 0.75};
    const auto selection = ml_scratch::select_threshold_for_f1(
        validation_targets, validation_probabilities, candidate_thresholds);

    std::vector<int> test_targets;
    std::vector<int> majority_predictions;
    for (const auto& sample : test_data) {
        test_targets.push_back(sample.target);
        majority_predictions.push_back(0);
    }
    const auto baseline = ml_scratch::classification_metrics(test_targets, majority_predictions);
    const auto default_metrics = model.evaluate(test_data, 0.5);
    const auto selected_metrics = model.evaluate(test_data, selection.threshold);

    std::cout << "C++ binary logistic regression experiment (seed=" << seed << ")\n"
              << std::fixed << std::setprecision(6) << "epochs=" << result.epochs
              << ", final weighted BCE=" << result.loss_per_epoch.back()
              << ", weight=" << model.weights()[0] << ", bias=" << model.bias() << '\n'
              << "validation-selected threshold=" << selection.threshold
              << ", validation F1=" << selection.metrics.f1 << '\n';
    print_metrics("majority baseline (test)", baseline);
    print_metrics("threshold 0.50 (test)", default_metrics);
    print_metrics("selected threshold (test)", selected_metrics);

    if (!result.converged || selection.metrics.f1 == 0.0 ||
        selected_metrics.f1 <= default_metrics.f1 || selected_metrics.f1 <= baseline.f1) {
        std::cerr << "model failed its convergence or held-out F1 criteria\n";
        return 1;
    }
    return 0;
}
