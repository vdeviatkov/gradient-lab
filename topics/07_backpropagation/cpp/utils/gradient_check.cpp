#include "ml_scratch/utils/gradient_check.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ml_scratch {

namespace {

template <typename Model, typename Objective>
std::vector<double> central_differences(const Model& network, Objective objective,
                                        const double epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("epsilon must be finite and positive");
    }
    static_cast<void>(objective(network));
    Model probe = network;
    const auto original = network.parameters();
    auto values = original;
    std::vector<double> numerical(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = original[index] + epsilon;
        probe.set_parameters(values);
        const double above = objective(probe);
        values[index] = original[index] - epsilon;
        probe.set_parameters(values);
        const double below = objective(probe);
        values[index] = original[index];
        numerical[index] = (above - below) / (2.0 * epsilon);
    }
    return numerical;
}

} // namespace

std::vector<double> numerical_gradient(const FeedForwardNetwork& network,
                                       const SupervisedDataset& dataset, const double epsilon) {
    return central_differences(
        network, [&](const FeedForwardNetwork& probe) { return probe.loss(dataset); }, epsilon);
}

GradientCheckResult check_gradient(const std::vector<double>& analytic,
                                   const std::vector<double>& numerical, const double tolerance) {
    if (analytic.empty()) {
        throw std::invalid_argument("gradients must not be empty");
    }
    if (analytic.size() != numerical.size()) {
        throw std::invalid_argument("gradients must have equal sizes");
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument("tolerance must be finite and positive");
    }

    GradientCheckResult result{0.0, 0.0, 0.0, 0, analytic.front(), numerical.front(), true};
    double difference_norm = 0.0;
    double analytic_norm = 0.0;
    double numerical_norm = 0.0;

    for (std::size_t index = 0; index < analytic.size(); ++index) {
        const double difference = std::abs(analytic[index] - numerical[index]);
        difference_norm += difference * difference;
        analytic_norm += analytic[index] * analytic[index];
        numerical_norm += numerical[index] * numerical[index];

        if (difference > result.max_absolute_error) {
            result.max_absolute_error = difference;
            result.worst_parameter = index;
            result.analytic_value = analytic[index];
            result.numerical_value = numerical[index];
        }
        // The floor keeps two entries that are both essentially zero from dividing a tiny
        // difference by an equally tiny denominator.
        const double denominator =
            std::max(std::abs(analytic[index]) + std::abs(numerical[index]), 1e-8);
        result.max_elementwise_relative_error =
            std::max(result.max_elementwise_relative_error, difference / denominator);
    }

    const double scale = std::sqrt(analytic_norm) + std::sqrt(numerical_norm);
    // An all-zero gradient pair is a perfect match, not a division by zero.
    result.relative_error = scale == 0.0 ? 0.0 : std::sqrt(difference_norm) / scale;
    result.passed = result.relative_error <= tolerance;
    return result;
}

GradientCheckResult check_gradient(const FeedForwardNetwork& network,
                                   const SupervisedDataset& dataset, const double epsilon,
                                   const double tolerance) {
    return check_gradient(network.gradient(dataset), numerical_gradient(network, dataset, epsilon),
                          tolerance);
}

GradientCheckResult check_batch_gradient(FeedForwardNetwork& network, const LabeledDataset& batch,
                                         const double epsilon, const double tolerance) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("epsilon must be finite and positive");
    }
    if (network.uses_dropout()) {
        throw std::invalid_argument(
            "gradient checking needs a deterministic loss, so dropout must be disabled");
    }

    FeedForwardNetwork probe = network;
    const std::vector<double> analytic = probe.batch_training_gradient(batch);
    const std::vector<double> original = network.parameters();
    std::vector<double> perturbed = original;
    std::vector<double> numerical(original.size(), 0.0);

    // The batch is perturbed as one unit, because batch normalization makes the loss a function of
    // the whole batch rather than a sum over independent samples.
    for (std::size_t index = 0; index < original.size(); ++index) {
        perturbed[index] = original[index] + epsilon;
        probe.set_parameters(perturbed);
        const double raised = probe.batch_training_loss(batch);

        perturbed[index] = original[index] - epsilon;
        probe.set_parameters(perturbed);
        const double lowered = probe.batch_training_loss(batch);

        perturbed[index] = original[index];
        numerical[index] = (raised - lowered) / (2.0 * epsilon);
    }
    probe.set_parameters(original);
    return check_gradient(analytic, numerical, tolerance);
}

std::vector<double> numerical_gradient(const CharRnn& network, const TokenSequence& ids,
                                       const CharRnn::State& state, const double epsilon) {
    return central_differences(
        network,
        [&](const CharRnn& probe) {
            auto running = state.empty() ? probe.initial_state() : state;
            return probe.loss(ids, running);
        },
        epsilon);
}

std::vector<double> numerical_gradient(const CharTransformer& network, const TokenSequence& ids,
                                       const double epsilon) {
    return central_differences(
        network, [&](const CharTransformer& probe) { return probe.loss(ids); }, epsilon);
}

} // namespace ml_scratch
