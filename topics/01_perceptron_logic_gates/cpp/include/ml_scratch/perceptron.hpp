#pragma once

#include "ml_scratch/logic_gates.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ml_scratch {

struct PerceptronTrainingResult {
    std::size_t epochs;
    bool converged;
    std::vector<std::size_t> mistakes_per_epoch;

    bool operator==(const PerceptronTrainingResult&) const = default;
};

class BinaryPerceptron {
  public:
    explicit BinaryPerceptron(double learning_rate = 0.1, std::uint32_t seed = 0);

    [[nodiscard]] double decision_score(const LogicInput& features) const noexcept;
    [[nodiscard]] int predict(const LogicInput& features) const noexcept;
    [[nodiscard]] double accuracy(const LogicDataset& dataset) const noexcept;

    PerceptronTrainingResult fit(LogicDataset dataset, std::size_t max_epochs = 100,
                                 bool shuffle = true);

    [[nodiscard]] const std::array<double, 2>& weights() const noexcept { return weights_; }
    [[nodiscard]] double bias() const noexcept { return bias_; }

  private:
    double learning_rate_;
    std::mt19937 random_engine_;
    std::array<double, 2> weights_{};
    double bias_{};
};

} // namespace ml_scratch
