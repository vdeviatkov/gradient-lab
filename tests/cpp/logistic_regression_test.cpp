#include "ml_scratch/logistic_regression.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

void require_near(const double actual, const double expected, const double tolerance,
                  const std::string_view message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

template <typename Function>
void require_invalid_argument(Function function, const std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

const ml_scratch::BinaryClassificationDataset separable_data{
    {{-3.0}, 0}, {{-2.0}, 0}, {{-1.0}, 0}, {{1.0}, 1}, {{2.0}, 1}, {{3.0}, 1},
};

void test_confusion_matrix_and_metrics() {
    const std::vector<int> targets{0, 0, 1, 1};
    const std::vector<int> predictions{0, 1, 1, 0};
    const auto metrics = ml_scratch::classification_metrics(targets, predictions);

    require(metrics.confusion == ml_scratch::ConfusionMatrix{1, 1, 1, 1},
            "incorrect confusion matrix");
    require_near(metrics.accuracy, 0.5, 1e-12, "incorrect accuracy");
    require_near(metrics.precision, 0.5, 1e-12, "incorrect precision");
    require_near(metrics.recall, 0.5, 1e-12, "incorrect recall");
    require_near(metrics.f1, 0.5, 1e-12, "incorrect F1");

    const auto no_positive_predictions = ml_scratch::classification_metrics({0, 0, 1}, {0, 0, 0});
    require(no_positive_predictions.precision == 0.0, "zero-denominator precision should be zero");
    require(no_positive_predictions.f1 == 0.0, "zero precision should produce zero F1");
}

void test_threshold_selection() {
    const auto selected = ml_scratch::select_threshold_for_f1(
        {1, 1, 0, 0}, {0.45, 0.40, 0.42, 0.10}, {0.50, 0.45, 0.40});
    require_near(selected.threshold, 0.40, 1e-12, "did not select the best F1 threshold");
    require_near(selected.metrics.f1, 0.8, 1e-12, "incorrect selected-threshold F1");

    const auto tie = ml_scratch::select_threshold_for_f1({0, 1}, {0.1, 0.9}, {0.3, 0.7, 0.5});
    require_near(tie.threshold, 0.5, 1e-12, "threshold tie did not prefer 0.5");
}

void test_training_and_numerical_stability() {
    ml_scratch::BinaryLogisticRegression model{1};
    require_near(model.binary_cross_entropy(separable_data), std::log(2.0), 1e-12,
                 "zero-parameter loss should equal log(2)");

    ml_scratch::LogisticTrainingConfig config;
    config.learning_rate = 0.2;
    config.max_epochs = 2'000;
    config.target_loss = 0.03;
    const auto result = model.fit(separable_data, config);

    require(result.converged, "logistic regression did not converge on separable data");
    require(result.epochs == result.loss_per_epoch.size(), "loss history has wrong size");
    require(result.loss_per_epoch.back() <= config.target_loss,
            "reported convergence above target loss");
    const auto metrics = model.evaluate(separable_data);
    require(metrics.accuracy == 1.0, "trained classifier missed a separable example");

    const double low_probability = model.predict_probability({-1e6});
    const double high_probability = model.predict_probability({1e6});
    require(std::isfinite(low_probability) && low_probability == 0.0,
            "stable sigmoid failed for a large negative score");
    require(std::isfinite(high_probability) && high_probability == 1.0,
            "stable sigmoid failed for a large positive score");
}

void test_multivariate_training() {
    const ml_scratch::BinaryClassificationDataset dataset{
        {{-2.0, 0.0}, 0}, {{0.0, -2.0}, 0}, {{-1.0, -1.0}, 0},
        {{2.0, 0.0}, 1},  {{0.0, 2.0}, 1},  {{1.0, 1.0}, 1},
    };
    ml_scratch::LogisticTrainingConfig config;
    config.learning_rate = 0.2;
    config.max_epochs = 2'000;
    config.target_loss = 0.03;

    ml_scratch::BinaryLogisticRegression model{2};
    const auto result = model.fit(dataset, config);
    require(result.converged, "multivariate logistic regression did not converge");
    require(model.evaluate(dataset).accuracy == 1.0,
            "multivariate logistic regression missed a training example");
    require(model.weights()[0] > 0.0 && model.weights()[1] > 0.0,
            "multivariate weights point in the wrong direction");
    require_near(model.weights()[0], model.weights()[1], 1e-12,
                 "symmetric features learned asymmetric weights");
}

void test_seed_reproducibility() {
    ml_scratch::LogisticTrainingConfig config;
    config.learning_rate = 0.05;
    config.max_epochs = 30;
    config.batch_size = 2;
    config.seed = 1618;

    ml_scratch::BinaryLogisticRegression first{1};
    ml_scratch::BinaryLogisticRegression second{1};
    const auto first_result = first.fit(separable_data, config);
    const auto second_result = second.fit(separable_data, config);
    require(first_result == second_result, "same seed produced different loss histories");
    require(first.weights() == second.weights(), "same seed produced different weights");
    require(first.bias() == second.bias(), "same seed produced different biases");
}

void test_class_imbalance_metrics() {
    const std::vector<int> targets{0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
    const std::vector<int> majority_predictions(10, 0);
    const std::vector<int> useful_predictions{0, 0, 0, 0, 0, 0, 0, 1, 1, 1};

    const auto majority = ml_scratch::classification_metrics(targets, majority_predictions);
    const auto useful = ml_scratch::classification_metrics(targets, useful_predictions);
    require_near(majority.accuracy, 0.8, 1e-12, "incorrect majority-baseline accuracy");
    require(majority.recall == 0.0 && majority.f1 == 0.0,
            "majority baseline unexpectedly detects positives");
    require(useful.accuracy > majority.accuracy && useful.f1 > majority.f1,
            "imbalance-aware metrics did not reward useful positive predictions");
}

void test_positive_class_weight() {
    const ml_scratch::BinaryClassificationDataset imbalanced{
        {{0.0}, 0}, {{0.0}, 0}, {{0.0}, 0}, {{0.0}, 1}};
    ml_scratch::LogisticTrainingConfig config;
    config.learning_rate = 0.1;
    config.max_epochs = 1'000;

    ml_scratch::BinaryLogisticRegression unweighted{1};
    static_cast<void>(unweighted.fit(imbalanced, config));
    require_near(unweighted.predict_probability({0.0}), 0.25, 1e-6,
                 "unweighted intercept did not learn the positive frequency");

    config.positive_class_weight = 3.0;
    ml_scratch::BinaryLogisticRegression weighted{1};
    static_cast<void>(weighted.fit(imbalanced, config));
    require_near(weighted.predict_probability({0.0}), 0.5, 1e-12,
                 "positive weighting did not balance the class contributions");
}

void test_input_validation() {
    require_invalid_argument([] { const ml_scratch::BinaryLogisticRegression model{0}; },
                             "zero feature count was accepted");
    require_invalid_argument([] { static_cast<void>(ml_scratch::classification_metrics({}, {})); },
                             "empty metrics input was accepted");
    require_invalid_argument(
        [] { static_cast<void>(ml_scratch::classification_metrics({0}, {0, 1})); },
        "mismatched metrics input was accepted");
    require_invalid_argument(
        [] { static_cast<void>(ml_scratch::classification_metrics({2}, {0})); },
        "non-binary target was accepted");
    require_invalid_argument(
        [] {
            static_cast<void>(
                ml_scratch::select_threshold_for_f1({1}, {0.5}, std::vector<double>{}));
        },
        "empty threshold candidates were accepted");

    ml_scratch::BinaryLogisticRegression model{1};
    require_invalid_argument([&] { static_cast<void>(model.predict({1.0}, 1.1)); },
                             "out-of-range threshold was accepted");
    require_invalid_argument([&] { static_cast<void>(model.predict({1.0, 2.0})); },
                             "wrong feature count was accepted");
    require_invalid_argument([&] { static_cast<void>(model.evaluate({{{1.0}, 2}})); },
                             "non-binary training target was accepted");
    require_invalid_argument(
        [&] {
            static_cast<void>(model.predict_probability({std::numeric_limits<double>::infinity()}));
        },
        "non-finite feature was accepted");

    ml_scratch::LogisticTrainingConfig config;
    config.learning_rate = 0.0;
    require_invalid_argument([&] { model.fit(separable_data, config); },
                             "zero learning rate was accepted");
    config.learning_rate = 0.1;
    config.max_epochs = 0;
    require_invalid_argument([&] { model.fit(separable_data, config); },
                             "zero epoch budget was accepted");
    config.max_epochs = 10;
    config.positive_class_weight = 0.0;
    require_invalid_argument([&] { model.fit(separable_data, config); },
                             "zero positive class weight was accepted");
}

} // namespace

int main() {
    try {
        test_confusion_matrix_and_metrics();
        test_threshold_selection();
        test_training_and_numerical_stability();
        test_multivariate_training();
        test_seed_reproducibility();
        test_class_imbalance_metrics();
        test_positive_class_weight();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
