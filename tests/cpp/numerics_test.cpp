#include "ml_scratch/activation.hpp"
#include "ml_scratch/numerics.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("shared numerical primitive contract failed");
}
void near(double actual, double expected, double tolerance = 1e-12) {
    require(std::abs(actual - expected) <= tolerance);
}
} // namespace

int main() {
    try {
        using namespace ml_scratch;
        near(sigmoid(0.0), 0.5);
        require(sigmoid(-1000.0) == 0.0 && sigmoid(1000.0) == 1.0);
        near(softplus(0.0), std::log(2.0));
        near(softplus(1000.0), 1000.0);
        require(softplus(-1000.0) == 0.0);
        const auto probabilities = softmax({1000.0, 1001.0, 999.0});
        const auto shifted = softmax({0.0, 1.0, -1.0});
        double total = 0.0;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            near(probabilities[i], shifted[i]);
            total += probabilities[i];
        }
        near(total, 1.0);
        for (auto kind : {Activation::identity, Activation::sigmoid, Activation::hyperbolic_tangent,
                          Activation::rectified_linear, Activation::leaky_rectified_linear}) {
            for (double x : {-0.7, 0.7}) {
                const double value = apply_activation(x, kind);
                constexpr double epsilon = 1e-6;
                const double numerical =
                    (apply_activation(x + epsilon, kind) - apply_activation(x - epsilon, kind)) /
                    (2 * epsilon);
                near(activation_derivative(x, value, kind), numerical, 1e-9);
            }
        }
        near(activation_derivative(0.0, 0.0, Activation::rectified_linear), 0.0);
        near(activation_derivative(0.0, 0.0, Activation::leaky_rectified_linear),
             leaky_rectified_linear_slope);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
