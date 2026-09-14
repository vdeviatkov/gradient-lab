#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/optimizer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

void require_near(const double actual, const double expected, const double tolerance,
                  const std::string_view message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

template <typename Function>
void require_invalid_argument(Function function, const std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

ml_scratch::Optimizer make(const ml_scratch::OptimizerKind kind, const std::size_t count = 1) {
    ml_scratch::OptimizerConfig config;
    config.kind = kind;
    return ml_scratch::Optimizer{config, count};
}

void test_gradient_descent_is_the_plain_rule() {
    auto optimizer = make(ml_scratch::OptimizerKind::gradient_descent);
    std::vector<double> update;

    optimizer.compute_update({2.0}, 0.1, update);
    require_near(update[0], 0.2, 1e-15, "gradient descent should scale the gradient by the rate");
    optimizer.compute_update({2.0}, 0.1, update);
    require_near(update[0], 0.2, 1e-15, "gradient descent must be stateless");
    require(optimizer.first_state().empty(), "gradient descent should allocate no state");
    require(optimizer.steps() == 2, "step counter is wrong");
}

void test_momentum_recursion() {
    auto optimizer = make(ml_scratch::OptimizerKind::momentum);
    std::vector<double> update;
    constexpr double rate = 0.1;
    constexpr double decay = 0.9;

    // v1 = g, v2 = mu g + g, v3 = mu^2 g + mu g + g for a constant gradient.
    double velocity = 0.0;
    for (std::size_t step = 1; step <= 4; ++step) {
        optimizer.compute_update({2.0}, rate, update);
        velocity = decay * velocity + 2.0;
        require_near(update[0], rate * velocity, 1e-15, "incorrect momentum update");
    }
    // The geometric series caps the effective step at lr * g / (1 - mu).
    require(update[0] < rate * 2.0 / (1.0 - decay), "momentum exceeded its asymptotic step");
    require(update[0] > rate * 2.0, "momentum did not accelerate past plain descent");

    optimizer.reset();
    optimizer.compute_update({2.0}, rate, update);
    require_near(update[0], rate * 2.0, 1e-15, "reset did not clear the velocity");
    require(optimizer.steps() == 1, "reset did not clear the step counter");
}

void test_nesterov_looks_ahead() {
    auto nesterov = make(ml_scratch::OptimizerKind::nesterov_momentum);
    auto classical = make(ml_scratch::OptimizerKind::momentum);
    std::vector<double> nesterov_update;
    std::vector<double> classical_update;

    // First step: v = g, so Nesterov takes lr(g + mu g) while classical momentum takes lr g.
    nesterov.compute_update({2.0}, 0.1, nesterov_update);
    classical.compute_update({2.0}, 0.1, classical_update);
    require_near(nesterov_update[0], 0.1 * (2.0 + 0.9 * 2.0), 1e-15, "incorrect Nesterov update");
    require(nesterov_update[0] > classical_update[0],
            "Nesterov should step further while the gradient is consistent");

    // On a sign flip the look-ahead term brakes harder than classical momentum.
    for (std::size_t step = 0; step < 5; ++step) {
        nesterov.compute_update({2.0}, 0.1, nesterov_update);
        classical.compute_update({2.0}, 0.1, classical_update);
    }
    nesterov.compute_update({-2.0}, 0.1, nesterov_update);
    classical.compute_update({-2.0}, 0.1, classical_update);
    require(nesterov_update[0] < classical_update[0],
            "Nesterov should respond to a reversed gradient sooner");
}

void test_rmsprop_normalizes_each_coordinate() {
    auto optimizer = make(ml_scratch::OptimizerKind::rmsprop, 2);
    std::vector<double> update;
    constexpr double rate = 0.1;
    constexpr double decay = 0.9;

    // First step: s = (1 - rho) g^2, so the step is lr g / (sqrt((1 - rho)) |g| + eps), which is
    // nearly lr / sqrt(1 - rho) times the sign of g regardless of the gradient's size.
    optimizer.compute_update({100.0, 0.01}, rate, update);
    const double expected_large = rate * 100.0 / (std::sqrt((1.0 - decay) * 100.0 * 100.0) + 1e-8);
    const double expected_small = rate * 0.01 / (std::sqrt((1.0 - decay) * 0.01 * 0.01) + 1e-8);
    require_near(update[0], expected_large, 1e-12, "incorrect RMSProp update");
    require_near(update[1], expected_small, 1e-12, "incorrect RMSProp update");
    // Gradients four orders of magnitude apart produce nearly equal steps.
    require_near(update[0] / update[1], 1.0, 1e-3,
                 "RMSProp did not equalize steps across very different gradient scales");

    // A coordinate whose gradient vanishes keeps a decaying memory of its past magnitude.
    optimizer.compute_update({0.0, 0.01}, rate, update);
    require_near(update[0], 0.0, 0.0, "a zero gradient should produce no step");
    require(optimizer.first_state()[0] > 0.0, "the squared-gradient average should persist");
}

void test_adam_bias_correction() {
    auto optimizer = make(ml_scratch::OptimizerKind::adam);
    std::vector<double> update;
    constexpr double rate = 0.01;

    // The point of bias correction: at step one the corrected moments are exactly g and g^2, so
    // the step is lr * g / (|g| + eps), essentially lr * sign(g) whatever the gradient's scale.
    optimizer.compute_update({5.0}, rate, update);
    require_near(update[0], rate * 5.0 / (5.0 + 1e-8), 1e-12, "incorrect first Adam step");
    require_near(update[0], rate, 1e-6, "Adam's first step should be about the learning rate");

    auto tiny = make(ml_scratch::OptimizerKind::adam);
    std::vector<double> tiny_update;
    tiny.compute_update({1e-6}, rate, tiny_update);
    require_near(tiny_update[0], rate, 1e-3,
                 "Adam should take a full-sized first step even for a tiny gradient");

    // Without correction the first step would be scaled by (1 - beta1) / sqrt(1 - beta2) ~ 3.2,
    // so the corrected step must differ from the raw-moment step by that factor.
    auto reference = make(ml_scratch::OptimizerKind::adam);
    std::vector<double> reference_update;
    reference.compute_update({5.0}, rate, reference_update);
    const double raw_first = 0.1 * 5.0;
    const double raw_second = 0.001 * 25.0;
    const double uncorrected = rate * raw_first / (std::sqrt(raw_second) + 1e-8);
    require(std::abs(reference_update[0] - uncorrected) > 1e-4,
            "bias correction had no effect on the first step");

    // The correction factors approach one, so late steps behave like uncorrected moments.
    for (std::size_t step = 0; step < 5'000; ++step) {
        optimizer.compute_update({5.0}, rate, update);
    }
    require_near(update[0], rate * 5.0 / (5.0 + 1e-8), 1e-6,
                 "a constant gradient should give a steady Adam step");
    require(optimizer.second_state()[0] > 0.0, "Adam should track a second moment");
}

// Descends f(x) = x^2 / 2 from x = 1 and returns the final distance to the minimum.
double descend_quadratic(const ml_scratch::OptimizerKind kind, const double learning_rate,
                         const std::size_t steps) {
    ml_scratch::OptimizerConfig config;
    config.kind = kind;
    ml_scratch::Optimizer optimizer{config, 1};
    std::vector<double> parameters{1.0};
    std::vector<double> update;
    for (std::size_t step = 0; step < steps; ++step) {
        // The gradient of x^2 / 2 is x.
        optimizer.compute_update({parameters[0]}, learning_rate, update);
        parameters[0] -= update[0];
    }
    return std::abs(parameters[0]);
}

// Every rule must make substantial progress on a convex quadratic from the same start.
void test_all_optimizers_descend_a_quadratic() {
    for (const ml_scratch::OptimizerKind kind :
         {ml_scratch::OptimizerKind::gradient_descent, ml_scratch::OptimizerKind::momentum,
          ml_scratch::OptimizerKind::nesterov_momentum, ml_scratch::OptimizerKind::rmsprop,
          ml_scratch::OptimizerKind::adam}) {
        require(descend_quadratic(kind, 0.01, 2'000) < 1e-2,
                std::string{"optimizer did not descend the quadratic: "} +
                    std::string{ml_scratch::optimizer_name(kind)});
    }
}

// RMSProp divides out the gradient's magnitude, so its step stays roughly lr * sign(g) however
// close the parameter gets. With a fixed learning rate it therefore settles into a limit cycle at a
// distance proportional to lr instead of converging, while the rules that keep a longer memory of
// past gradients contract all the way. This is the concrete reason a decaying learning rate
// matters, and it is asserted rather than described.
void test_rmsprop_hovers_while_others_contract() {
    // On this problem the limit cycle sits at exactly lr / 2, so the residual is predictable and
    // proportional to the learning rate rather than merely nonzero.
    for (const double rate : {0.02, 0.01, 0.005}) {
        const double residual = descend_quadratic(ml_scratch::OptimizerKind::rmsprop, rate, 2'000);
        require(residual > 1e-4, "RMSProp unexpectedly contracted to the minimum");
        require_near(residual / (rate / 2.0), 1.0, 1e-4,
                     "the RMSProp limit cycle was not at half the learning rate");
    }

    for (const ml_scratch::OptimizerKind kind :
         {ml_scratch::OptimizerKind::gradient_descent, ml_scratch::OptimizerKind::momentum,
          ml_scratch::OptimizerKind::nesterov_momentum, ml_scratch::OptimizerKind::adam}) {
        require(descend_quadratic(kind, 0.01, 2'000) < 1e-8,
                std::string{"expected contraction to the minimum from "} +
                    std::string{ml_scratch::optimizer_name(kind)});
    }
}

// On an ill-conditioned quadratic the per-coordinate rules should beat a single global step size.
void test_adaptive_rules_handle_poor_conditioning() {
    // f(x, y) = (a x^2 + b y^2) / 2 with a curvature ratio of 1000.
    constexpr double steep = 100.0;
    constexpr double flat = 0.1;
    const auto final_distance = [](const ml_scratch::OptimizerKind kind) {
        ml_scratch::OptimizerConfig config;
        config.kind = kind;
        ml_scratch::Optimizer optimizer{config, 2};
        std::vector<double> parameters{1.0, 1.0};
        std::vector<double> update;
        for (std::size_t step = 0; step < 500; ++step) {
            // The largest stable rate for plain descent here is 2/steep, so 0.01 is near its limit.
            optimizer.compute_update({steep * parameters[0], flat * parameters[1]}, 0.01, update);
            parameters[0] -= update[0];
            parameters[1] -= update[1];
        }
        return std::sqrt(parameters[0] * parameters[0] + parameters[1] * parameters[1]);
    };

    const double plain = final_distance(ml_scratch::OptimizerKind::gradient_descent);
    const double adam = final_distance(ml_scratch::OptimizerKind::adam);
    const double rmsprop = final_distance(ml_scratch::OptimizerKind::rmsprop);

    // Plain descent is limited by the steep direction and barely moves along the flat one.
    require(plain > 0.5, "plain descent unexpectedly solved the ill-conditioned problem");
    require(adam < plain, "Adam did not beat plain descent on a poorly conditioned quadratic");
    require(rmsprop < plain,
            "RMSProp did not beat plain descent on a poorly conditioned quadratic");
}

void test_network_integration() {
    const ml_scratch::SupervisedDataset xor_data{
        {{0.0, 0.0}, {0.0}},
        {{0.0, 1.0}, {1.0}},
        {{1.0, 0.0}, {1.0}},
        {{1.0, 1.0}, {0.0}},
    };
    const std::vector<ml_scratch::DenseLayer> layers{
        {2, 4, ml_scratch::Activation::hyperbolic_tangent},
        {4, 1, ml_scratch::Activation::identity},
    };

    for (const ml_scratch::OptimizerKind kind :
         {ml_scratch::OptimizerKind::gradient_descent, ml_scratch::OptimizerKind::momentum,
          ml_scratch::OptimizerKind::nesterov_momentum, ml_scratch::OptimizerKind::rmsprop,
          ml_scratch::OptimizerKind::adam}) {
        ml_scratch::NetworkTrainingConfig config;
        config.optimizer.kind = kind;
        // The adaptive rules normalize the step, so they need a smaller rate than plain descent.
        const bool adaptive =
            kind == ml_scratch::OptimizerKind::rmsprop || kind == ml_scratch::OptimizerKind::adam;
        config.learning_rate = adaptive ? 0.01 : 0.1;
        config.max_epochs = 20'000;
        config.target_loss = 0.02;

        ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::binary_cross_entropy,
                                               20260912};
        const auto result = network.fit(xor_data, config);
        require(result.converged, std::string{"XOR did not converge with "} +
                                      std::string{ml_scratch::optimizer_name(kind)});
        require(ml_scratch::check_gradient(network, xor_data).passed,
                "the gradient check failed after training with an optimizer");
    }

    // The default configuration must remain plain gradient descent, so earlier milestones that
    // never mention an optimizer keep their exact behavior.
    ml_scratch::NetworkTrainingConfig defaults;
    require(defaults.optimizer.kind == ml_scratch::OptimizerKind::gradient_descent,
            "the default optimizer changed");
}

void test_determinism_and_validation() {
    auto first = make(ml_scratch::OptimizerKind::adam, 3);
    auto second = make(ml_scratch::OptimizerKind::adam, 3);
    std::vector<double> first_update;
    std::vector<double> second_update;
    for (std::size_t step = 0; step < 10; ++step) {
        const std::vector<double> gradient{0.5, -1.5, 0.25};
        first.compute_update(gradient, 0.01, first_update);
        second.compute_update(gradient, 0.01, second_update);
    }
    require(first_update == second_update, "the same inputs produced different updates");

    require_invalid_argument([] { make(ml_scratch::OptimizerKind::adam, 0); },
                             "a zero parameter count was accepted");
    require_invalid_argument(
        [] {
            ml_scratch::OptimizerConfig config;
            config.momentum = 1.0;
            return ml_scratch::Optimizer{config, 1};
        },
        "a momentum of one was accepted");
    require_invalid_argument(
        [] {
            ml_scratch::OptimizerConfig config;
            config.beta2 = -0.1;
            return ml_scratch::Optimizer{config, 1};
        },
        "a negative beta2 was accepted");
    require_invalid_argument(
        [] {
            ml_scratch::OptimizerConfig config;
            config.epsilon = 0.0;
            return ml_scratch::Optimizer{config, 1};
        },
        "a zero epsilon was accepted");

    auto optimizer = make(ml_scratch::OptimizerKind::adam, 2);
    std::vector<double> update;
    require_invalid_argument([&] { optimizer.compute_update({1.0}, 0.1, update); },
                             "a mismatched gradient size was accepted");
    require_invalid_argument([&] { optimizer.compute_update({1.0, 2.0}, 0.0, update); },
                             "a zero learning rate was accepted");

    require(ml_scratch::optimizer_name(ml_scratch::OptimizerKind::adam) == "Adam",
            "incorrect optimizer name");
}

} // namespace

int main() {
    try {
        test_gradient_descent_is_the_plain_rule();
        test_momentum_recursion();
        test_nesterov_looks_ahead();
        test_rmsprop_normalizes_each_coordinate();
        test_adam_bias_correction();
        test_all_optimizers_descend_a_quadratic();
        test_rmsprop_hovers_while_others_contract();
        test_adaptive_rules_handle_poor_conditioning();
        test_network_integration();
        test_determinism_and_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
