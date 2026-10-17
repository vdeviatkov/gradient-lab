#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace ml_scratch {

enum class OptimizerKind {
    // theta <- theta - lr * g
    gradient_descent,
    // Accumulates a velocity, so consistent gradient directions build up speed.
    momentum,
    // Momentum evaluated as if the velocity step had already been taken.
    nesterov_momentum,
    // Divides each coordinate by the square root of a decaying average of its squared gradients.
    rmsprop,
    // RMSProp's per-coordinate scaling combined with momentum, both bias-corrected.
    adam,
};

[[nodiscard]] std::string_view optimizer_name(OptimizerKind kind);

struct OptimizerConfig {
    OptimizerKind kind{OptimizerKind::gradient_descent};
    // Velocity decay for momentum and nesterov_momentum.
    double momentum{0.9};
    // Decay of the squared-gradient average for rmsprop.
    double decay_rate{0.9};
    // Adam's first- and second-moment decays.
    double beta1{0.9};
    double beta2{0.999};
    // Floor on the per-coordinate denominator of rmsprop and adam.
    double epsilon{1e-8};

    bool operator==(const OptimizerConfig&) const = default;
};

// Turns a gradient into a parameter update. The optimizer owns only its own state and
// hyperparameters; the learning rate is supplied per step, so a caller can apply a schedule without
// the optimizer knowing about it.
class Optimizer {
  public:
    Optimizer(OptimizerConfig config, std::size_t parameter_count);

    // Writes the amount to SUBTRACT from each parameter into `update`, which is resized as needed.
    // Separating this from the parameters themselves lets a model keep whatever internal layout it
    // likes while the update rules stay written once, against a flat vector.
    void compute_update(const std::vector<double>& gradient, double learning_rate,
                        std::vector<double>& update);

    // Clears the accumulated state and the step counter, leaving the hyperparameters alone.
    void reset();

    [[nodiscard]] std::size_t steps() const noexcept { return steps_; }
    [[nodiscard]] const OptimizerConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }
    // Empty for gradient_descent; the velocity for momentum; the squared-gradient average for
    // rmsprop; the first moment for adam.
    [[nodiscard]] const std::vector<double>& first_state() const noexcept { return first_moment_; }
    // The second moment, used by adam only.
    [[nodiscard]] const std::vector<double>& second_state() const noexcept {
        return second_moment_;
    }

  private:
    OptimizerConfig config_;
    std::size_t parameter_count_;
    std::size_t steps_{0};
    std::vector<double> first_moment_;
    std::vector<double> second_moment_;
};

} // namespace ml_scratch
