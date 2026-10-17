#include "ml_scratch/perceptron.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ml_scratch {

double BinaryPerceptron::accuracy(const LogicDataset& dataset) const noexcept {
    std::size_t correct = 0;
    for (const auto& [features, target] : dataset) {
        correct += predict(features) == target ? 1U : 0U;
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

} // namespace ml_scratch
