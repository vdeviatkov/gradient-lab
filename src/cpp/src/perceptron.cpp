#include "ml_scratch/perceptron.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ml_scratch {

BinaryPerceptron::BinaryPerceptron(const double learning_rate, const std::uint32_t seed)
    : learning_rate_{learning_rate}, random_engine_{seed} {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be positive");
    }

    std::uniform_real_distribution<double> initial_value{-0.5, 0.5};
    for (double& weight : weights_) {
        weight = initial_value(random_engine_);
    }
    bias_ = initial_value(random_engine_);
}

double BinaryPerceptron::decision_score(const LogicInput& features) const noexcept {
    return weights_[0] * features[0] + weights_[1] * features[1] + bias_;
}

int BinaryPerceptron::predict(const LogicInput& features) const noexcept {
    return decision_score(features) >= 0.0 ? 1 : 0;
}

double BinaryPerceptron::accuracy(const LogicDataset& dataset) const noexcept {
    std::size_t correct = 0;
    for (const auto& [features, target] : dataset) {
        correct += predict(features) == target ? 1U : 0U;
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

PerceptronTrainingResult BinaryPerceptron::fit(LogicDataset dataset, const std::size_t max_epochs,
                                               const bool shuffle) {
    if (max_epochs == 0) {
        throw std::invalid_argument("max_epochs must be positive");
    }

    std::vector<std::size_t> history;
    history.reserve(max_epochs);

    for (std::size_t epoch = 1; epoch <= max_epochs; ++epoch) {
        if (shuffle) {
            std::shuffle(dataset.begin(), dataset.end(), random_engine_);
        }

        std::size_t mistakes = 0;
        for (const auto& [features, target] : dataset) {
            const int error = target - predict(features);
            if (error == 0) {
                continue;
            }

            const double adjustment = learning_rate_ * static_cast<double>(error);
            for (std::size_t index = 0; index < weights_.size(); ++index) {
                weights_[index] += adjustment * features[index];
            }
            bias_ += adjustment;
            ++mistakes;
        }

        history.push_back(mistakes);
        if (mistakes == 0) {
            return {epoch, true, std::move(history)};
        }
    }

    return {max_epochs, false, std::move(history)};
}

} // namespace ml_scratch
