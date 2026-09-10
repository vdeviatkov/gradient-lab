#include "ml_scratch/dataset.hpp"
#include "ml_scratch/deterministic_random.hpp"
#include "ml_scratch/neural_network.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t seed = 20260910;
constexpr std::uint32_t network_seed = 20260910;
constexpr std::size_t class_count = 3;
constexpr std::size_t samples_per_class = 150;
// Splits are interleaved rather than taken as index ranges: the index sweeps the position along a
// spiral arm, so a range split would hand each split a disjoint piece of the arm and measure
// extrapolation instead of generalization. Three of every five samples train, then one each for
// validation and test.
constexpr std::size_t split_period = 5;

struct Splits {
    ml_scratch::SupervisedDataset train;
    ml_scratch::SupervisedDataset validation;
    ml_scratch::SupervisedDataset test;
};

// Three interleaved spiral arms. The classes are separable but not linearly separable, so a linear
// model is a meaningful baseline and hidden layers have to do real work.
Splits build_spiral() {
    ml_scratch::DeterministicRandom random{seed};
    Splits splits;
    for (std::size_t label = 0; label < class_count; ++label) {
        for (std::size_t index = 0; index < samples_per_class; ++index) {
            const double position =
                static_cast<double>(index) / static_cast<double>(samples_per_class - 1);
            const double radius = 0.2 + 0.8 * position;
            const double angle = 4.0 * position +
                                 2.0 * 3.14159265358979323846 * static_cast<double>(label) /
                                     static_cast<double>(class_count) +
                                 0.14 * random.gaussian();

            std::vector<double> features{radius * std::sin(angle), radius * std::cos(angle)};
            std::vector<double> targets(class_count, 0.0);
            targets[label] = 1.0;

            const std::size_t position_in_period = index % split_period;
            ml_scratch::SupervisedDataset& split = position_in_period < 3    ? splits.train
                                                   : position_in_period == 3 ? splits.validation
                                                                             : splits.test;
            split.push_back({std::move(features), std::move(targets)});
        }
    }
    return splits;
}

struct Configuration {
    std::string_view name;
    std::vector<ml_scratch::DenseLayer> layers;
    ml_scratch::Loss loss;
};

const ml_scratch::SupervisedDataset regression_data{
    {{0.3, -0.7}, {0.4, -0.2}},
    {{-0.5, 0.9}, {-0.1, 0.8}},
    {{0.8, 0.2}, {0.6, 0.1}},
};

const ml_scratch::SupervisedDataset binary_data{
    {{0.0, 0.0}, {0.0}},
    {{0.0, 1.0}, {1.0}},
    {{1.0, 0.0}, {1.0}},
    {{1.0, 1.0}, {0.0}},
};

// A one-sided difference, implemented here rather than in the library because its only purpose is
// to show why the library uses the central form.
std::vector<double> forward_difference_gradient(const ml_scratch::FeedForwardNetwork& network,
                                                const ml_scratch::SupervisedDataset& dataset,
                                                const double epsilon) {
    ml_scratch::FeedForwardNetwork probe = network;
    const std::vector<double> original = network.parameters();
    std::vector<double> perturbed = original;
    std::vector<double> estimate(original.size(), 0.0);
    const double base = probe.loss(dataset);

    for (std::size_t index = 0; index < original.size(); ++index) {
        perturbed[index] = original[index] + epsilon;
        probe.set_parameters(perturbed);
        estimate[index] = (probe.loss(dataset) - base) / epsilon;
        perturbed[index] = original[index];
    }
    return estimate;
}

double accuracy_of_majority_baseline(const Splits& splits) {
    std::vector<std::size_t> counts(class_count, 0);
    for (const auto& sample : splits.train) {
        for (std::size_t label = 0; label < class_count; ++label) {
            if (sample.targets[label] == 1.0) {
                ++counts[label];
            }
        }
    }
    std::size_t majority = 0;
    for (std::size_t label = 1; label < class_count; ++label) {
        if (counts[label] > counts[majority]) {
            majority = label;
        }
    }

    std::size_t correct = 0;
    for (const auto& sample : splits.test) {
        if (sample.targets[majority] == 1.0) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(splits.test.size());
}

} // namespace

int main() {
    const Splits splits = build_spiral();
    const ml_scratch::SupervisedDataset& gradient_data = splits.train;

    std::cout << "C++ backpropagation and gradient checking experiment (seed=" << seed << ")\n"
              << "spiral dataset: " << class_count * samples_per_class << " samples, 2 features, "
              << class_count << " classes; splits train=" << splits.train.size()
              << ", validation=" << splits.validation.size() << ", test=" << splits.test.size()
              << "\n\n";

    const std::vector<Configuration> configurations{
        {"linear, squared error",
         {{2, 2, ml_scratch::Activation::identity}},
         ml_scratch::Loss::mean_squared_error},
        {"tanh hidden, squared error",
         {{2, 5, ml_scratch::Activation::hyperbolic_tangent},
          {5, 2, ml_scratch::Activation::identity}},
         ml_scratch::Loss::mean_squared_error},
        {"sigmoid output, squared error",
         {{2, 4, ml_scratch::Activation::hyperbolic_tangent},
          {4, 2, ml_scratch::Activation::sigmoid}},
         ml_scratch::Loss::mean_squared_error},
        {"binary cross-entropy",
         {{2, 4, ml_scratch::Activation::hyperbolic_tangent},
          {4, 1, ml_scratch::Activation::identity}},
         ml_scratch::Loss::binary_cross_entropy},
        {"softmax cross-entropy",
         {{2, 8, ml_scratch::Activation::hyperbolic_tangent},
          {8, 3, ml_scratch::Activation::identity}},
         ml_scratch::Loss::softmax_cross_entropy},
        {"three hidden layers, mixed",
         {{2, 6, ml_scratch::Activation::hyperbolic_tangent},
          {6, 6, ml_scratch::Activation::sigmoid},
          {6, 4, ml_scratch::Activation::hyperbolic_tangent},
          {4, 3, ml_scratch::Activation::identity}},
         ml_scratch::Loss::softmax_cross_entropy},
        {"ReLU hidden layers",
         {{2, 6, ml_scratch::Activation::rectified_linear},
          {6, 6, ml_scratch::Activation::rectified_linear},
          {6, 3, ml_scratch::Activation::identity}},
         ml_scratch::Loss::softmax_cross_entropy},
    };

    std::cout
        << "gradient check at initialization (central differences, epsilon=1e-5, tolerance=1e-7)\n";
    bool every_check_passed = true;
    for (const Configuration& configuration : configurations) {
        const ml_scratch::SupervisedDataset& data =
            configuration.loss == ml_scratch::Loss::binary_cross_entropy ? binary_data
            : configuration.loss == ml_scratch::Loss::mean_squared_error ? regression_data
                                                                         : gradient_data;
        ml_scratch::FeedForwardNetwork network{configuration.layers, configuration.loss,
                                               network_seed};
        const auto result = ml_scratch::check_gradient(network, data);
        every_check_passed = every_check_passed && result.passed;

        std::string label{configuration.name};
        label.resize(30, ' ');
        std::cout << "  " << label << " parameters=" << std::setw(4) << network.parameter_count()
                  << ", gradient-check ratio=" << std::scientific << std::setprecision(3)
                  << result.relative_error << std::defaultfloat
                  << (result.passed ? "  pass" : "  FAIL") << '\n';
    }

    // The step size trades two error sources against each other. Truncation falls as epsilon^2,
    // while cancellation in the difference of two nearly equal losses grows as 1/epsilon.
    ml_scratch::FeedForwardNetwork probe{{{2, 8, ml_scratch::Activation::hyperbolic_tangent},
                                          {8, 3, ml_scratch::Activation::identity}},
                                         ml_scratch::Loss::softmax_cross_entropy,
                                         network_seed};
    const std::vector<double> analytic = probe.gradient(gradient_data);

    std::cout << "\nstep-size sweep on the softmax network (" << probe.parameter_count()
              << " parameters)\n"
              << "  epsilon    central ratio   forward ratio\n";
    double best_central = 1.0;
    double best_epsilon = 0.0;
    double central_at_1e5 = 1.0;
    double forward_at_1e5 = 1.0;
    for (const double epsilon : {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10}) {
        const double central =
            ml_scratch::check_gradient(analytic, probe.numerical_gradient(gradient_data, epsilon))
                .relative_error;
        const double forward =
            ml_scratch::check_gradient(analytic,
                                       forward_difference_gradient(probe, gradient_data, epsilon))
                .relative_error;
        std::cout << "  " << std::scientific << std::setprecision(0) << std::setw(8) << epsilon
                  << std::setprecision(3) << std::setw(16) << central << std::setw(16) << forward
                  << std::defaultfloat << '\n';
        if (central < best_central) {
            best_central = central;
            best_epsilon = epsilon;
        }
        if (epsilon == 1e-5) {
            central_at_1e5 = central;
            forward_at_1e5 = forward;
        }
    }
    std::cout << "  best central-difference error " << std::scientific << std::setprecision(3)
              << best_central << " at epsilon=" << std::setprecision(0) << best_epsilon
              << std::defaultfloat << '\n';

    // A checker that never fails proves nothing, so corrupt a known-good gradient.
    std::vector<double> corrupted = analytic;
    corrupted[7] *= 1.001;
    const auto detection =
        ml_scratch::check_gradient(corrupted, probe.numerical_gradient(gradient_data, 1e-5));
    std::cout << "\ndetection of a deliberate 0.1% error in parameter 7: "
              << (detection.passed ? "MISSED" : "caught") << ", ratio=" << std::scientific
              << std::setprecision(3) << detection.relative_error << std::defaultfloat
              << " at parameter " << detection.worst_parameter << '\n';

    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.5;
    config.max_epochs = 4'000;
    config.batch_size = 32;
    config.seed = network_seed;

    ml_scratch::FeedForwardNetwork linear{{{2, 3, ml_scratch::Activation::identity}},
                                          ml_scratch::Loss::softmax_cross_entropy,
                                          network_seed};
    const auto linear_result = linear.fit(splits.train, config);

    ml_scratch::FeedForwardNetwork mlp{{{2, 16, ml_scratch::Activation::hyperbolic_tangent},
                                        {16, 16, ml_scratch::Activation::hyperbolic_tangent},
                                        {16, 3, ml_scratch::Activation::identity}},
                                       ml_scratch::Loss::softmax_cross_entropy,
                                       network_seed};
    const auto initial_check = ml_scratch::check_gradient(mlp, splits.train);

    // A short run first, so the gradient can also be checked away from both initialization and the
    // eventual minimum.
    ml_scratch::NetworkTrainingConfig warmup = config;
    warmup.max_epochs = 20;
    static_cast<void>(mlp.fit(splits.train, warmup));
    const std::vector<double> partial_gradient = mlp.gradient(splits.train);
    const auto partial_check = ml_scratch::check_gradient(mlp, splits.train);

    const auto mlp_result = mlp.fit(splits.train, config);
    const auto converged_check = ml_scratch::check_gradient(mlp, splits.train);

    const double baseline = accuracy_of_majority_baseline(splits);
    std::cout << std::fixed << std::setprecision(4) << "\ntraining on the spiral ("
              << config.max_epochs << " epochs, batch=" << config.batch_size
              << ", learning rate=" << config.learning_rate << ")\n"
              << "  majority baseline           : test accuracy=" << baseline << '\n'
              << "  linear softmax (" << std::setw(3) << linear.parameter_count()
              << " params) : final loss=" << linear_result.loss_per_epoch.back()
              << ", train=" << linear.accuracy(splits.train)
              << ", validation=" << linear.accuracy(splits.validation)
              << ", test=" << linear.accuracy(splits.test) << '\n'
              << "  2x16 tanh MLP (" << std::setw(3) << mlp.parameter_count()
              << " params) : final loss=" << mlp_result.loss_per_epoch.back()
              << ", train=" << mlp.accuracy(splits.train)
              << ", validation=" << mlp.accuracy(splits.validation)
              << ", test=" << mlp.accuracy(splits.test) << '\n';

    // Relative error is only meaningful while the gradient itself is not vanishing. At a minimum
    // every entry approaches zero, so a fixed round-off floor dominates the ratio even though the
    // two gradients still agree to several significant figures.
    std::cout << "\ngradient check of the MLP at three points on its trajectory\n"
              << "  point            norm ratio   max elementwise   max absolute   "
                 "largest |gradient|\n";
    const auto report_check = [](const std::string_view name,
                                 const ml_scratch::GradientCheckResult& check,
                                 const std::vector<double>& gradient) {
        double largest = 0.0;
        for (const double value : gradient) {
            largest = std::max(largest, std::abs(value));
        }
        std::string label{name};
        label.resize(17, ' ');
        std::cout << "  " << label << std::scientific << std::setprecision(3) << std::setw(11)
                  << check.relative_error << std::setw(18) << check.max_elementwise_relative_error
                  << std::setw(15) << check.max_absolute_error << std::setw(21) << largest
                  << std::defaultfloat << (check.passed ? "  pass" : "  FAIL") << '\n';
    };
    ml_scratch::FeedForwardNetwork fresh{mlp.layers(), mlp.loss_function(), network_seed};
    report_check("initialization", initial_check, fresh.gradient(splits.train));
    report_check("after 20 epochs", partial_check, partial_gradient);
    report_check("at convergence", converged_check, mlp.gradient(splits.train));

    const bool central_beats_forward = central_at_1e5 < forward_at_1e5;
    const bool error_detected = !detection.passed;
    const bool mlp_beat_linear = mlp.accuracy(splits.test) > linear.accuracy(splits.test);
    const bool linear_beat_baseline = linear.accuracy(splits.test) > baseline;

    // At convergence the relative check is expected to degrade; the absolute agreement is what
    // still carries information there.
    const bool converged_gradients_still_agree = converged_check.max_absolute_error < 1e-9;

    if (!every_check_passed || !central_beats_forward || !error_detected || !mlp_beat_linear ||
        !linear_beat_baseline || !initial_check.passed || !partial_check.passed ||
        !converged_gradients_still_agree) {
        std::cerr << "experiment failed its criteria\n";
        return 1;
    }
    return 0;
}
