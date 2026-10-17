#include "ml_scratch/gradient.hpp"

#include <cmath>
#include <stdexcept>

namespace ml_scratch {

double clip_gradient(std::vector<double>& gradient, const double max_norm) {
    // An infinite threshold is allowed: it never clips but still measures the norm.
    if (std::isnan(max_norm) || max_norm <= 0.0) {
        throw std::invalid_argument("the clipping threshold must be positive");
    }
    double squared = 0.0;
    for (const double value : gradient) {
        squared += value * value;
    }
    const double norm = std::sqrt(squared);
    if (norm > max_norm) {
        const double factor = max_norm / norm;
        for (double& value : gradient) {
            value *= factor;
        }
    }
    return norm;
}

} // namespace ml_scratch
