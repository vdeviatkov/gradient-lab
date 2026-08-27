#include "ml_scratch/xor_mlp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace ml_scratch {
namespace {

double sigmoid(const double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

} // namespace

XorMlp::XorMlp(const std::size_t hidden_units, const double learning_rate, const std::uint32_t seed)
    : learning_rate_{learning_rate},
      parameters_{std::vector<std::array<double, 2>>(hidden_units),
                  std::vector<double>(hidden_units, 0.0), std::vector<double>(hidden_units), 0.0} {
    if (hidden_units == 0) {
        throw std::invalid_argument("hidden_units must be positive");
    }
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    std::mt19937 random_engine{seed};
    const double hidden_limit = std::sqrt(6.0 / (2.0 + static_cast<double>(hidden_units)));
    const double output_limit = std::sqrt(6.0 / (static_cast<double>(hidden_units) + 1.0));
    std::uniform_real_distribution<double> hidden_initial_value{-hidden_limit, hidden_limit};
    std::uniform_real_distribution<double> output_initial_value{-output_limit, output_limit};

    for (auto& weights : parameters_.input_weights) {
        weights = {hidden_initial_value(random_engine), hidden_initial_value(random_engine)};
    }
    for (double& weight : parameters_.output_weights) {
        weight = output_initial_value(random_engine);
    }
}

std::vector<double> XorMlp::hidden_activations(const LogicInput& features) const {
    std::vector<double> activations(hidden_units());
    for (std::size_t hidden = 0; hidden < hidden_units(); ++hidden) {
        const auto& weights = parameters_.input_weights[hidden];
        const double score =
            weights[0] * features[0] + weights[1] * features[1] + parameters_.hidden_biases[hidden];
        activations[hidden] = std::tanh(score);
    }
    return activations;
}

double XorMlp::predict_probability(const LogicInput& features) const {
    const std::vector<double> hidden_values = hidden_activations(features);
    double output_score = parameters_.output_bias;
    for (std::size_t hidden = 0; hidden < hidden_units(); ++hidden) {
        output_score += parameters_.output_weights[hidden] * hidden_values[hidden];
    }
    return sigmoid(output_score);
}

int XorMlp::predict(const LogicInput& features) const {
    return predict_probability(features) >= 0.5 ? 1 : 0;
}

double XorMlp::accuracy(const LogicDataset& dataset) const {
    std::size_t correct = 0;
    for (const auto& [features, target] : dataset) {
        correct += predict(features) == target ? 1U : 0U;
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

double XorMlp::binary_cross_entropy(const LogicDataset& dataset) const {
    double total_loss = 0.0;
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (const auto& [features, target] : dataset) {
        const double probability =
            std::clamp(predict_probability(features), epsilon, 1.0 - epsilon);
        const double expected = static_cast<double>(target);
        total_loss -=
            expected * std::log(probability) + (1.0 - expected) * std::log(1.0 - probability);
    }
    return total_loss / static_cast<double>(dataset.size());
}

XorMlpTrainingResult XorMlp::fit(const LogicDataset& dataset, const std::size_t max_epochs,
                                 const double target_loss) {
    if (max_epochs == 0) {
        throw std::invalid_argument("max_epochs must be positive");
    }
    if (target_loss <= 0.0) {
        throw std::invalid_argument("target_loss must be positive");
    }

    std::vector<double> loss_history;
    loss_history.reserve(max_epochs);
    const double inverse_sample_count = 1.0 / static_cast<double>(dataset.size());

    for (std::size_t epoch = 1; epoch <= max_epochs; ++epoch) {
        std::vector<std::array<double, 2>> input_weight_gradients(hidden_units(), {0.0, 0.0});
        std::vector<double> hidden_bias_gradients(hidden_units(), 0.0);
        std::vector<double> output_weight_gradients(hidden_units(), 0.0);
        double output_bias_gradient = 0.0;

        for (const auto& [features, target] : dataset) {
            const std::vector<double> hidden_values = hidden_activations(features);
            double output_score = parameters_.output_bias;
            for (std::size_t hidden = 0; hidden < hidden_units(); ++hidden) {
                output_score += parameters_.output_weights[hidden] * hidden_values[hidden];
            }

            // For sigmoid plus binary cross-entropy, d(loss)/d(output_score) = p - y.
            const double output_delta = sigmoid(output_score) - static_cast<double>(target);
            output_bias_gradient += output_delta;

            for (std::size_t hidden = 0; hidden < hidden_units(); ++hidden) {
                output_weight_gradients[hidden] += output_delta * hidden_values[hidden];
                const double hidden_delta = output_delta * parameters_.output_weights[hidden] *
                                            (1.0 - hidden_values[hidden] * hidden_values[hidden]);
                hidden_bias_gradients[hidden] += hidden_delta;
                input_weight_gradients[hidden][0] += hidden_delta * features[0];
                input_weight_gradients[hidden][1] += hidden_delta * features[1];
            }
        }

        for (std::size_t hidden = 0; hidden < hidden_units(); ++hidden) {
            parameters_.output_weights[hidden] -=
                learning_rate_ * output_weight_gradients[hidden] * inverse_sample_count;
            parameters_.hidden_biases[hidden] -=
                learning_rate_ * hidden_bias_gradients[hidden] * inverse_sample_count;
            for (std::size_t input = 0; input < 2; ++input) {
                parameters_.input_weights[hidden][input] -=
                    learning_rate_ * input_weight_gradients[hidden][input] * inverse_sample_count;
            }
        }
        parameters_.output_bias -= learning_rate_ * output_bias_gradient * inverse_sample_count;

        const double loss = binary_cross_entropy(dataset);
        loss_history.push_back(loss);
        if (accuracy(dataset) == 1.0 && loss <= target_loss) {
            return {epoch, true, std::move(loss_history)};
        }
    }

    return {max_epochs, false, std::move(loss_history)};
}

} // namespace ml_scratch
