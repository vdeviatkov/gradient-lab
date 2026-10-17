#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace ml_scratch {

// Stable shared scalar primitives. softmax requires a nonempty vector of finite logits.
inline double sigmoid(const double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

inline double softplus(const double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

inline std::vector<double> softmax(const std::vector<double>& logits) {
    const double largest = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(logits.size());
    double total = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(logits[index] - largest);
        total += probabilities[index];
    }
    for (double& probability : probabilities) {
        probability /= total;
    }
    return probabilities;
}

} // namespace ml_scratch
