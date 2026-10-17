#pragma once

#include "ml_scratch/numerics.hpp"

namespace ml_scratch {

enum class Activation {
    identity,
    sigmoid,
    hyperbolic_tangent,
    // Not differentiable at zero. Gradient checking can report a spurious mismatch for a
    // pre-activation that lands on the kink; see topics/07_backpropagation/idea.md.
    rectified_linear,
    // max(v, slope * v): a rectifier whose negative side keeps a small fixed slope, so no unit can
    // stop passing gradient by landing in the flat region. A GAN's discriminator uses it because
    // the generator learns only through the gradient the discriminator passes back.
    leaky_rectified_linear,
};

// The negative-side slope of leaky_rectified_linear, the value the DCGAN paper uses.
inline constexpr double leaky_rectified_linear_slope = 0.2;

inline double apply_activation(const double value, const Activation activation) {
    switch (activation) {
    case Activation::sigmoid:
        return sigmoid(value);
    case Activation::hyperbolic_tangent:
        return std::tanh(value);
    case Activation::rectified_linear:
        return std::max(value, 0.0);
    case Activation::leaky_rectified_linear:
        return value > 0.0 ? value : leaky_rectified_linear_slope * value;
    case Activation::identity:
        break;
    }
    return value;
}

inline double activation_derivative(const double pre_activation, const double activation_value,
                                    const Activation activation) {
    switch (activation) {
    case Activation::sigmoid:
        return activation_value * (1.0 - activation_value);
    case Activation::hyperbolic_tangent:
        return 1.0 - activation_value * activation_value;
    case Activation::rectified_linear:
        // The kink at zero has no derivative; the subgradient zero is the usual convention.
        return pre_activation > 0.0 ? 1.0 : 0.0;
    case Activation::leaky_rectified_linear:
        return pre_activation > 0.0 ? 1.0 : leaky_rectified_linear_slope;
    case Activation::identity:
        break;
    }
    return 1.0;
}

} // namespace ml_scratch
