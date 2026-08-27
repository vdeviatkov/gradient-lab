#pragma once

#include "ml_scratch/logic_gates.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ml_scratch {

struct XorMlpTrainingResult {
    std::size_t epochs;
    bool converged;
    std::vector<double> loss_per_epoch;

    bool operator==(const XorMlpTrainingResult&) const = default;
};

struct XorMlpParameters {
    std::vector<std::array<double, 2>> input_weights;
    std::vector<double> hidden_biases;
    std::vector<double> output_weights;
    double output_bias;

    bool operator==(const XorMlpParameters&) const = default;
};

class XorMlp {
  public:
    explicit XorMlp(std::size_t hidden_units = 4, double learning_rate = 0.5,
                    std::uint32_t seed = 0);

    [[nodiscard]] double predict_probability(const LogicInput& features) const;
    [[nodiscard]] int predict(const LogicInput& features) const;
    [[nodiscard]] double accuracy(const LogicDataset& dataset) const;
    [[nodiscard]] double binary_cross_entropy(const LogicDataset& dataset) const;

    XorMlpTrainingResult fit(const LogicDataset& dataset, std::size_t max_epochs = 10'000,
                             double target_loss = 0.02);

    [[nodiscard]] const XorMlpParameters& parameters() const noexcept { return parameters_; }
    [[nodiscard]] std::size_t hidden_units() const noexcept {
        return parameters_.hidden_biases.size();
    }

  private:
    [[nodiscard]] std::vector<double> hidden_activations(const LogicInput& features) const;

    double learning_rate_;
    XorMlpParameters parameters_;
};

} // namespace ml_scratch
