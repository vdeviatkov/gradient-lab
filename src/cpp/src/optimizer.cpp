#include "ml_scratch/optimizer.hpp"

#include <cmath>
#include <stdexcept>

namespace ml_scratch {

std::string_view optimizer_name(const OptimizerKind kind) {
    switch (kind) {
    case OptimizerKind::gradient_descent:
        return "gradient descent";
    case OptimizerKind::momentum:
        return "momentum";
    case OptimizerKind::nesterov_momentum:
        return "Nesterov momentum";
    case OptimizerKind::rmsprop:
        return "RMSProp";
    case OptimizerKind::adam:
        return "Adam";
    }
    return "unknown";
}

Optimizer::Optimizer(OptimizerConfig config, const std::size_t parameter_count)
    : config_(config), parameter_count_(parameter_count) {
    if (parameter_count == 0) {
        throw std::invalid_argument("parameter_count must be positive");
    }
    const auto require_unit_interval = [](const double value, const char* name) {
        if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
            throw std::invalid_argument(std::string{name} +
                                        " must be finite and in the interval [0, 1)");
        }
    };
    require_unit_interval(config_.momentum, "momentum");
    require_unit_interval(config_.decay_rate, "decay_rate");
    require_unit_interval(config_.beta1, "beta1");
    require_unit_interval(config_.beta2, "beta2");
    if (!std::isfinite(config_.epsilon) || config_.epsilon <= 0.0) {
        throw std::invalid_argument("epsilon must be finite and positive");
    }

    // Plain gradient descent keeps no state, so it allocates none.
    if (config_.kind != OptimizerKind::gradient_descent) {
        first_moment_.assign(parameter_count, 0.0);
    }
    if (config_.kind == OptimizerKind::adam) {
        second_moment_.assign(parameter_count, 0.0);
    }
}

void Optimizer::reset() {
    steps_ = 0;
    first_moment_.assign(first_moment_.size(), 0.0);
    second_moment_.assign(second_moment_.size(), 0.0);
}

void Optimizer::compute_update(const std::vector<double>& gradient, const double learning_rate,
                               std::vector<double>& update) {
    if (gradient.size() != parameter_count_) {
        throw std::invalid_argument("gradient size does not match the optimizer");
    }
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }

    update.resize(parameter_count_);
    ++steps_;

    switch (config_.kind) {
    case OptimizerKind::gradient_descent:
        for (std::size_t index = 0; index < parameter_count_; ++index) {
            update[index] = learning_rate * gradient[index];
        }
        return;

    case OptimizerKind::momentum:
        // v <- mu v + g, step = lr v. A coordinate whose gradient keeps its sign reaches an
        // effective step of lr/(1 - mu) times the gradient, which is why momentum crosses long
        // shallow valleys faster than plain descent.
        for (std::size_t index = 0; index < parameter_count_; ++index) {
            first_moment_[index] = config_.momentum * first_moment_[index] + gradient[index];
            update[index] = learning_rate * first_moment_[index];
        }
        return;

    case OptimizerKind::nesterov_momentum:
        // The velocity is updated first and the step is taken from the look-ahead point, which
        // damps the overshoot classical momentum shows when it approaches a minimum at speed.
        for (std::size_t index = 0; index < parameter_count_; ++index) {
            first_moment_[index] = config_.momentum * first_moment_[index] + gradient[index];
            update[index] =
                learning_rate * (gradient[index] + config_.momentum * first_moment_[index]);
        }
        return;

    case OptimizerKind::rmsprop:
        // s <- rho s + (1 - rho) g^2, step = lr g / (sqrt(s) + eps). Dividing by the typical
        // magnitude of each coordinate's gradient makes the step size scale-free per coordinate.
        for (std::size_t index = 0; index < parameter_count_; ++index) {
            first_moment_[index] = config_.decay_rate * first_moment_[index] +
                                   (1.0 - config_.decay_rate) * gradient[index] * gradient[index];
            update[index] = learning_rate * gradient[index] /
                            (std::sqrt(first_moment_[index]) + config_.epsilon);
        }
        return;

    case OptimizerKind::adam: {
        // Both moments start at zero, so early averages are biased toward zero. Dividing by
        // 1 - beta^t removes exactly that bias; without it the first steps would be far too small.
        const double first_correction = 1.0 - std::pow(config_.beta1, static_cast<double>(steps_));
        const double second_correction = 1.0 - std::pow(config_.beta2, static_cast<double>(steps_));
        for (std::size_t index = 0; index < parameter_count_; ++index) {
            first_moment_[index] =
                config_.beta1 * first_moment_[index] + (1.0 - config_.beta1) * gradient[index];
            second_moment_[index] = config_.beta2 * second_moment_[index] +
                                    (1.0 - config_.beta2) * gradient[index] * gradient[index];
            const double corrected_first = first_moment_[index] / first_correction;
            const double corrected_second = second_moment_[index] / second_correction;
            update[index] =
                learning_rate * corrected_first / (std::sqrt(corrected_second) + config_.epsilon);
        }
        return;
    }
    }
    throw std::invalid_argument("unknown optimizer kind");
}

} // namespace ml_scratch
