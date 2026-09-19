#include "ml_scratch/deterministic_random.hpp"
#include "ml_scratch/gan.hpp"
#include "ml_scratch/neural_network.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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

template <typename Function>
void require_runtime_error(Function function, const std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, message);
}

double norm(const std::vector<double>& values) {
    double total = 0.0;
    for (const double value : values) {
        total += value * value;
    }
    return std::sqrt(total);
}

// Central differences of `function` over `parameters`, the same estimate the network's own
// numerical_gradient uses, written here so it can differentiate a loss the network does not own.
template <typename Function>
std::vector<double> numerical_gradient(std::vector<double> parameters, Function function,
                                       const double epsilon = 1e-5) {
    std::vector<double> gradient(parameters.size(), 0.0);
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const double original = parameters[index];
        parameters[index] = original + epsilon;
        const double above = function(parameters);
        parameters[index] = original - epsilon;
        const double below = function(parameters);
        parameters[index] = original;
        gradient[index] = (above - below) / (2.0 * epsilon);
    }
    return gradient;
}

// ---------------------------------------------------------------------------------------------
// FeedForwardNetwork additions the GAN relies on.
// ---------------------------------------------------------------------------------------------

void test_leaky_rectified_linear() {
    // One unit, weight 1, bias 0: the output is the activation of the input itself.
    ml_scratch::FeedForwardNetwork network{{{1, 1, ml_scratch::Activation::leaky_rectified_linear}},
                                           ml_scratch::Loss::mean_squared_error,
                                           1};
    network.set_parameters({1.0, 0.0});
    require_near(network.forward({2.0}).front(), 2.0, 1e-15,
                 "a positive input should pass through unchanged");
    require_near(network.forward({-2.0}).front(), -2.0 * ml_scratch::leaky_rectified_linear_slope,
                 1e-15, "a negative input should be scaled by the leak slope");
    require(ml_scratch::leaky_rectified_linear_slope > 0.0 &&
                ml_scratch::leaky_rectified_linear_slope < 1.0,
            "the leak must be a small positive slope");

    // The gradient on the negative side is the slope, not zero: that is the point of the leak.
    const ml_scratch::SupervisedDataset negative{{{-1.0}, {0.0}}};
    const auto gradient = network.gradient(negative);
    // L = (slope * (-1) - 0)^2, dL/dw = 2 * (slope * -1) * slope * (-1) = 2 slope^2.
    const double slope = ml_scratch::leaky_rectified_linear_slope;
    require_near(gradient[0], 2.0 * slope * slope, 1e-12,
                 "the weight gradient through a negative pre-activation should carry the slope");

    // And it survives the full gradient check inside a deeper network.
    const ml_scratch::SupervisedDataset data{
        {{0.3, -0.7}, {1.0, 0.0}}, {{-1.1, 0.4}, {0.0, 1.0}}, {{0.9, 0.9}, {1.0, 1.0}}};
    ml_scratch::FeedForwardNetwork deep{{{2, 5, ml_scratch::Activation::leaky_rectified_linear},
                                         {5, 4, ml_scratch::Activation::leaky_rectified_linear},
                                         {4, 2, ml_scratch::Activation::identity}},
                                        ml_scratch::Loss::binary_cross_entropy,
                                        7};
    require(ml_scratch::check_gradient(deep, data).passed,
            "the leaky rectifier failed its gradient check");
}

void test_input_gradient_matches_finite_differences() {
    // A network exercising every per-sample feature the backward pass has to carry to its input:
    // a skip, layer normalization, and each activation.
    const std::vector<ml_scratch::DenseLayer> layers{
        {3, 4, ml_scratch::Activation::hyperbolic_tangent},
        {4, 4, ml_scratch::Activation::leaky_rectified_linear, ml_scratch::Normalization::layer,
         0.0, true},
        {4, 4, ml_scratch::Activation::sigmoid, ml_scratch::Normalization::none, 0.0, true},
        {4, 2, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::binary_cross_entropy, 11};
    const std::vector<double> input{0.4, -0.9, 1.3};
    const std::vector<double> targets{1.0, 0.0};

    ml_scratch::FeedForwardNetwork::Workspace workspace;
    network.forward(input, workspace);
    std::vector<double> flat(network.parameter_count(), 0.0);
    std::vector<double> input_gradient;
    network.backpropagate_loss(targets, 1.0, &flat, workspace, &input_gradient);
    require(input_gradient.size() == input.size(), "the input gradient has the input's size");

    // The parameter gradient is the same one `gradient` computes on a one-sample dataset.
    const auto reference = network.gradient(ml_scratch::SupervisedDataset{{input, targets}});
    for (std::size_t index = 0; index < flat.size(); ++index) {
        require_near(flat[index], reference[index], 1e-14,
                     "backpropagate_loss disagrees with gradient on the parameters");
    }

    // The input gradient is checked against central differences of the loss over the input.
    const auto numerical = numerical_gradient(input, [&](const std::vector<double>& probe) {
        return network.loss(ml_scratch::SupervisedDataset{{probe, targets}});
    });
    const auto verdict = ml_scratch::check_gradient(input_gradient, numerical);
    require(verdict.passed, "the input gradient failed its finite-difference check");

    // Asking for the input gradient alone, with no parameter buffer, gives the same answer.
    std::vector<double> alone;
    network.forward(input, workspace);
    network.backpropagate_loss(targets, 1.0, nullptr, workspace, &alone);
    for (std::size_t index = 0; index < alone.size(); ++index) {
        require_near(alone[index], input_gradient[index], 0.0,
                     "the input gradient should not depend on whether parameters were requested");
    }
}

void test_backpropagate_takes_an_external_output_gradient() {
    // Under mean squared error dL/d(output) is 2 (a - t) / n; feeding exactly that gradient in
    // from outside must reproduce the network's own gradient, including the chain through the
    // sigmoid output activation.
    ml_scratch::FeedForwardNetwork network{{{2, 3, ml_scratch::Activation::hyperbolic_tangent},
                                            {3, 2, ml_scratch::Activation::sigmoid}},
                                           ml_scratch::Loss::mean_squared_error,
                                           5};
    const std::vector<double> input{0.25, -0.5};
    const std::vector<double> targets{0.9, 0.1};

    ml_scratch::FeedForwardNetwork::Workspace workspace;
    const std::vector<double> output = network.forward(input, workspace);
    std::vector<double> output_gradient(output.size());
    for (std::size_t index = 0; index < output.size(); ++index) {
        output_gradient[index] =
            2.0 * (output[index] - targets[index]) / static_cast<double>(output.size());
    }
    std::vector<double> flat(network.parameter_count(), 0.0);
    std::vector<double> input_gradient;
    network.backpropagate(output_gradient, 1.0, &flat, workspace, &input_gradient);

    const auto reference = network.gradient(ml_scratch::SupervisedDataset{{input, targets}});
    for (std::size_t index = 0; index < flat.size(); ++index) {
        require_near(flat[index], reference[index], 1e-14,
                     "an external MSE gradient should reproduce the network's own");
    }
    std::vector<double> reference_input;
    network.forward(input, workspace);
    network.backpropagate_loss(targets, 1.0, nullptr, workspace, &reference_input);
    for (std::size_t index = 0; index < reference_input.size(); ++index) {
        require_near(input_gradient[index], reference_input[index], 1e-14,
                     "the two backward entry points disagree on the input gradient");
    }

    // The scale multiplies the parameter gradient and accumulates onto what is already there.
    std::vector<double> accumulated(network.parameter_count(), 1.0);
    network.forward(input, workspace);
    network.backpropagate(output_gradient, 0.5, &accumulated, workspace);
    for (std::size_t index = 0; index < accumulated.size(); ++index) {
        require_near(accumulated[index], 1.0 + 0.5 * reference[index], 1e-14,
                     "backpropagate should scale and accumulate");
    }

    // A backward pass needs a forward pass into the same workspace first.
    ml_scratch::FeedForwardNetwork::Workspace fresh;
    bool rejected = false;
    try {
        network.backpropagate(output_gradient, 1.0, &flat, fresh);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "a backward pass without a forward pass should be refused");
    network.forward(input, workspace);
    require_invalid_argument([&] { network.backpropagate({1.0}, 1.0, &flat, workspace); },
                             "a wrongly sized output gradient should be rejected");
    std::vector<double> short_buffer(3, 0.0);
    require_invalid_argument(
        [&] { network.backpropagate(output_gradient, 1.0, &short_buffer, workspace); },
        "a wrongly sized parameter buffer should be rejected");
}

void test_stream_checkpoints_share_a_stream() {
    ml_scratch::FeedForwardNetwork first{{{2, 3, ml_scratch::Activation::leaky_rectified_linear},
                                          {3, 1, ml_scratch::Activation::identity}},
                                         ml_scratch::Loss::binary_cross_entropy,
                                         3};
    ml_scratch::FeedForwardNetwork second{{{1, 2, ml_scratch::Activation::hyperbolic_tangent,
                                            ml_scratch::Normalization::layer},
                                           {2, 2, ml_scratch::Activation::sigmoid}},
                                          ml_scratch::Loss::mean_squared_error,
                                          4};
    std::stringstream stream;
    first.save(stream, "first");
    second.save(stream, "second");

    const auto restored_first = ml_scratch::FeedForwardNetwork::load(stream, "first");
    const auto restored_second = ml_scratch::FeedForwardNetwork::load(stream, "second");
    require(restored_first.layers() == first.layers() && restored_second.layers() == second.layers(),
            "two networks written to one stream should both come back");
    require(restored_first.parameters() == first.parameters() &&
                restored_second.parameters() == second.parameters(),
            "stream checkpoints should round-trip every parameter exactly");
    require(restored_first.layers().front().activation ==
                ml_scratch::Activation::leaky_rectified_linear,
            "the leaky activation should survive a checkpoint");
}

// ---------------------------------------------------------------------------------------------
// ConditionalGan
// ---------------------------------------------------------------------------------------------

constexpr std::size_t noise_size = 3;
constexpr std::size_t class_count = 2;
constexpr std::size_t sample_size = 4;

std::vector<ml_scratch::DenseLayer> small_generator() {
    return {{noise_size + class_count, 6, ml_scratch::Activation::rectified_linear},
            {6, sample_size, ml_scratch::Activation::sigmoid}};
}

std::vector<ml_scratch::DenseLayer> small_discriminator() {
    return {{sample_size + class_count, 5, ml_scratch::Activation::leaky_rectified_linear},
            {5, 1, ml_scratch::Activation::identity}};
}

ml_scratch::FeatureMatrix some_noise(const std::size_t count, const std::uint64_t seed) {
    ml_scratch::DeterministicRandom random{seed};
    ml_scratch::FeatureMatrix noise(count, std::vector<double>(noise_size, 0.0));
    for (auto& row : noise) {
        for (double& value : row) {
            value = random.gaussian();
        }
    }
    return noise;
}

void test_construction_validation() {
    require_invalid_argument(
        [] {
            ml_scratch::ConditionalGan gan{
                noise_size, class_count,
                {{noise_size, 6, ml_scratch::Activation::rectified_linear},
                 {6, sample_size, ml_scratch::Activation::sigmoid}},
                small_discriminator()};
        },
        "a generator that leaves no room for the label should be rejected");
    require_invalid_argument(
        [] {
            ml_scratch::ConditionalGan gan{
                noise_size, class_count, small_generator(),
                {{sample_size, 5, ml_scratch::Activation::leaky_rectified_linear},
                 {5, 1, ml_scratch::Activation::identity}}};
        },
        "a discriminator that leaves no room for the label should be rejected");
    require_invalid_argument(
        [] {
            ml_scratch::ConditionalGan gan{
                noise_size, class_count, small_generator(),
                {{sample_size + class_count, 5, ml_scratch::Activation::leaky_rectified_linear},
                 {5, 2, ml_scratch::Activation::identity}}};
        },
        "a discriminator with two outputs should be rejected");
    require_invalid_argument(
        [] {
            ml_scratch::ConditionalGan gan{
                noise_size, class_count, small_generator(),
                {{sample_size + class_count, 5, ml_scratch::Activation::leaky_rectified_linear},
                 {5, 1, ml_scratch::Activation::sigmoid}}};
        },
        "a discriminator whose output is not a logit should be rejected");
    require_invalid_argument(
        [] {
            ml_scratch::ConditionalGan gan{
                noise_size, class_count, small_generator(),
                {{sample_size + class_count, 5, ml_scratch::Activation::leaky_rectified_linear,
                  ml_scratch::Normalization::batch},
                 {5, 1, ml_scratch::Activation::identity}}};
        },
        "batch normalization should be refused");
    require_invalid_argument(
        [] {
            ml_scratch::ConditionalGan gan{
                noise_size, class_count,
                {{noise_size + class_count, 6, ml_scratch::Activation::rectified_linear,
                  ml_scratch::Normalization::none, 0.5},
                 {6, sample_size, ml_scratch::Activation::sigmoid}},
                small_discriminator()};
        },
        "dropout should be refused");
    require_invalid_argument(
        [] {
            ml_scratch::ConditionalGan gan{0, class_count, small_generator(),
                                           small_discriminator()};
        },
        "an empty noise vector should be rejected");

    const ml_scratch::ConditionalGan gan{noise_size, class_count, small_generator(),
                                         small_discriminator(), 1};
    require(gan.noise_size() == noise_size && gan.class_count() == class_count &&
                gan.sample_size() == sample_size,
            "the sizes should be reported back");
    require_invalid_argument([&] { static_cast<void>(gan.generate({0.0, 0.0}, 0)); },
                             "wrongly sized noise should be rejected");
    require_invalid_argument([&] { static_cast<void>(gan.generate({0.0, 0.0, 0.0}, class_count)); },
                             "a label outside the class count should be rejected");
    require_invalid_argument([&] { static_cast<void>(gan.discriminate({0.0, 0.0}, 0)); },
                             "a wrongly sized sample should be rejected");
    ml_scratch::GanTrainingConfig bad;
    bad.batch_size = 0;
    ml_scratch::ConditionalGan trainee{noise_size, class_count, small_generator(),
                                       small_discriminator(), 1};
    const ml_scratch::LabeledDataset data{{{0.1, 0.2, 0.3, 0.4}, 0}};
    require_invalid_argument([&] { static_cast<void>(trainee.fit(data, bad)); },
                             "a zero batch size should be rejected");
    bad = {};
    bad.real_target = 0.0;
    require_invalid_argument([&] { static_cast<void>(trainee.fit(data, bad)); },
                             "a zero real target should be rejected");
    require_invalid_argument([&] { static_cast<void>(trainee.fit({{{0.1, 0.2}, 0}}, {})); },
                             "samples of the wrong size should be rejected");
    require_invalid_argument([&] { static_cast<void>(trainee.fit({{{0.1, 0.2, 0.3, 0.4}, 2}}, {})); },
                             "a dataset with more classes than the model should be rejected");
}

void test_generation_is_deterministic_and_conditioned() {
    const ml_scratch::ConditionalGan gan{noise_size, class_count, small_generator(),
                                         small_discriminator(), 21};
    std::mt19937 engine{5};
    const auto noise = gan.sample_noise(engine);
    require(noise.size() == noise_size, "noise has the configured size");
    std::mt19937 again{5};
    require(gan.sample_noise(again) == noise, "the same engine state gives the same noise");

    const auto sample = gan.generate(noise, 0);
    require(sample.size() == sample_size, "a sample has the configured size");
    for (const double value : sample) {
        require(value >= 0.0 && value <= 1.0, "a sigmoid generator stays inside [0, 1]");
    }
    require(gan.generate(noise, 0) == sample, "the same noise and label give the same sample");
    require(gan.generate(noise, 1) != sample, "the label should change the sample");

    const double probability = gan.discriminate(sample, 0);
    require(probability > 0.0 && probability < 1.0, "the discriminator returns a probability");

    // The noise is standard normal: over many draws its mean and variance say so.
    std::mt19937 many{99};
    double sum = 0.0;
    double sum_of_squares = 0.0;
    constexpr std::size_t draws = 20'000;
    for (std::size_t index = 0; index < draws; ++index) {
        for (const double value : gan.sample_noise(many)) {
            sum += value;
            sum_of_squares += value * value;
        }
    }
    const double count = static_cast<double>(draws * noise_size);
    const double mean = sum / count;
    const double variance = sum_of_squares / count - mean * mean;
    require_near(mean, 0.0, 0.02, "noise should have mean zero");
    require_near(variance, 1.0, 0.03, "noise should have unit variance");
}

void test_generator_gradient_matches_finite_differences() {
    ml_scratch::ConditionalGan gan{noise_size, class_count, small_generator(),
                                   small_discriminator(), 8};
    const auto noise = some_noise(6, 41);
    const std::vector<std::size_t> labels{0, 1, 1, 0, 0, 1};

    for (const auto loss : {ml_scratch::GeneratorLoss::non_saturating,
                            ml_scratch::GeneratorLoss::minimax}) {
        const auto analytic = gan.generator_gradient(noise, labels, loss);
        require(analytic.size() == gan.generator().parameter_count(),
                "the generator gradient covers the generator's parameters");
        // The generator's loss as a function of its parameters alone, with the discriminator held
        // fixed: exactly what the analytic gradient claims to differentiate.
        const auto numerical = numerical_gradient(
            gan.generator().parameters(), [&](const std::vector<double>& parameters) {
                gan.generator().set_parameters(parameters);
                return gan.generator_loss(noise, labels, loss);
            });
        const auto verdict = ml_scratch::check_gradient(analytic, numerical);
        require(verdict.passed, std::string{"the generator gradient failed its check under the "} +
                                    ml_scratch::generator_loss_name(loss) + " loss");
    }
}

void test_discriminator_gradient_matches_finite_differences() {
    ml_scratch::ConditionalGan gan{noise_size, class_count, small_generator(),
                                   small_discriminator(), 8};
    const ml_scratch::LabeledDataset real{
        {{0.9, 0.1, 0.8, 0.2}, 0}, {{0.8, 0.2, 0.9, 0.1}, 0}, {{0.1, 0.9, 0.2, 0.8}, 1}};
    const auto noise = some_noise(4, 17);
    const std::vector<std::size_t> labels{1, 0, 1, 0};

    for (const double real_target : {1.0, 0.9}) {
        const auto analytic = gan.discriminator_gradient(real, noise, labels, real_target);
        require(analytic.size() == gan.discriminator().parameter_count(),
                "the discriminator gradient covers the discriminator's parameters");
        const auto numerical = numerical_gradient(
            gan.discriminator().parameters(), [&](const std::vector<double>& parameters) {
                gan.discriminator().set_parameters(parameters);
                return gan.discriminator_loss(real, noise, labels, real_target);
            });
        require(ml_scratch::check_gradient(analytic, numerical).passed,
                "the discriminator gradient failed its check");
    }

    // The loss is what it says: mean cross-entropy toward the real target plus mean toward zero.
    double expected = 0.0;
    for (const auto& sample : real) {
        expected += -std::log(gan.discriminate(sample.features, sample.label)) /
                    static_cast<double>(real.size());
    }
    for (std::size_t index = 0; index < noise.size(); ++index) {
        const double fake = gan.discriminate(gan.generate(noise[index], labels[index]),
                                             labels[index]);
        expected += -std::log(1.0 - fake) / static_cast<double>(noise.size());
    }
    require_near(gan.discriminator_loss(real, noise, labels), expected, 1e-12,
                 "the discriminator loss should be the two cross-entropies");
}

// The reason the non-saturating loss exists. When the discriminator confidently rejects a fake,
// the minimax generator gradient is proportional to D(G(z)) and vanishes; the non-saturating one
// is proportional to 1 - D(G(z)) and is at its largest.
void test_minimax_loss_saturates_where_the_other_does_not() {
    ml_scratch::ConditionalGan gan{noise_size, class_count, small_generator(),
                                   small_discriminator(), 8};
    // Force the discriminator to reject everything: zero weights and a strongly negative bias.
    std::vector<double> parameters(gan.discriminator().parameter_count(), 0.0);
    parameters.back() = -8.0;
    gan.discriminator().set_parameters(parameters);
    const auto noise = some_noise(4, 3);
    const std::vector<std::size_t> labels{0, 1, 0, 1};
    const double rejection = gan.discriminate(gan.generate(noise[0], 0), 0);
    require(rejection < 0.001, "the discriminator should now reject every fake");

    // Zero weights in the discriminator's output layer would stop every gradient, so give the
    // hidden-to-output weights a value while keeping the bias in charge of the verdict.
    for (std::size_t index = 0; index + 1 < parameters.size(); ++index) {
        parameters[index] = 0.1;
    }
    gan.discriminator().set_parameters(parameters);
    const double non_saturating =
        norm(gan.generator_gradient(noise, labels, ml_scratch::GeneratorLoss::non_saturating));
    const double minimax =
        norm(gan.generator_gradient(noise, labels, ml_scratch::GeneratorLoss::minimax));
    require(non_saturating > 0.0, "the non-saturating gradient should be alive");
    // The two gradients differ exactly by the factor D(G(z)) / (1 - D(G(z))) per sample, which is
    // tiny when the discriminator is confident.
    require(minimax < 0.01 * non_saturating,
            "the minimax gradient should have all but vanished against a confident critic");
}

// A conditional distribution small enough to learn in a test: class 0 lives near one corner of
// the unit square and class 1 near the opposite one. After training, the generator's samples for
// each label must land near that label's corner, which is what "conditional" is supposed to buy.
void test_training_learns_a_conditional_distribution() {
    constexpr std::size_t toy_noise = 2;
    constexpr std::size_t toy_sample = 2;
    ml_scratch::DeterministicRandom random{2026};
    ml_scratch::LabeledDataset real;
    for (std::size_t index = 0; index < 400; ++index) {
        const std::size_t label = index % 2;
        const double center = label == 0 ? 0.2 : 0.8;
        real.push_back({{center + 0.05 * random.gaussian(), center + 0.05 * random.gaussian()},
                        label});
    }

    ml_scratch::ConditionalGan gan{
        toy_noise,
        class_count,
        {{toy_noise + class_count, 16, ml_scratch::Activation::leaky_rectified_linear},
         {16, toy_sample, ml_scratch::Activation::sigmoid}},
        {{toy_sample + class_count, 16, ml_scratch::Activation::leaky_rectified_linear},
         {16, 1, ml_scratch::Activation::identity}},
        12};

    // Before training, the generator has no idea where either class lives.
    const auto class_mean = [&](const std::size_t label) {
        std::vector<double> mean(toy_sample, 0.0);
        std::mt19937 fixed{100 + static_cast<std::uint32_t>(label)};
        constexpr std::size_t count = 200;
        for (std::size_t index = 0; index < count; ++index) {
            const auto sample = gan.generate(gan.sample_noise(fixed), label);
            for (std::size_t axis = 0; axis < toy_sample; ++axis) {
                mean[axis] += sample[axis] / static_cast<double>(count);
            }
        }
        return mean;
    };
    const auto before_zero = class_mean(0);
    const auto before_one = class_mean(1);

    ml_scratch::GanTrainingConfig config;
    config.learning_rate = 0.002;
    config.epochs = 60;
    config.batch_size = 32;
    config.seed = 3;
    const auto result = gan.fit(real, config);
    require(result.epochs == config.epochs &&
                result.discriminator_loss_per_epoch.size() == config.epochs &&
                result.generator_loss_per_epoch.size() == config.epochs &&
                result.real_score_per_epoch.size() == config.epochs &&
                result.fake_score_per_epoch.size() == config.epochs &&
                result.generator_gradient_norm_per_epoch.size() == config.epochs,
            "the result should hold one entry per epoch");
    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        require(std::isfinite(result.discriminator_loss_per_epoch[epoch]) &&
                    std::isfinite(result.generator_loss_per_epoch[epoch]),
                "losses should stay finite");
        require(result.real_score_per_epoch[epoch] > 0.0 && result.real_score_per_epoch[epoch] < 1.0 &&
                    result.fake_score_per_epoch[epoch] > 0.0 &&
                    result.fake_score_per_epoch[epoch] < 1.0,
                "scores are probabilities");
    }

    const auto after_zero = class_mean(0);
    const auto after_one = class_mean(1);
    for (std::size_t axis = 0; axis < toy_sample; ++axis) {
        require_near(after_zero[axis], 0.2, 0.1, "class 0 samples should sit near 0.2");
        require_near(after_one[axis], 0.8, 0.1, "class 1 samples should sit near 0.8");
        require(std::abs(after_zero[axis] - 0.2) < std::abs(before_zero[axis] - 0.2) &&
                    std::abs(after_one[axis] - 0.8) < std::abs(before_one[axis] - 0.8),
                "training should have moved each class toward its corner");
    }

    // Samples are not all the same point: the generator has to use its noise, or it has
    // collapsed. The real data's per-axis spread is 0.05, so a collapsed generator would show
    // something far below that.
    std::mt19937 spread_engine{55};
    double sum = 0.0;
    double sum_of_squares = 0.0;
    constexpr std::size_t count = 200;
    for (std::size_t index = 0; index < count; ++index) {
        const double value = gan.generate(gan.sample_noise(spread_engine), 0).front();
        sum += value;
        sum_of_squares += value * value;
    }
    const double mean = sum / count;
    const double deviation = std::sqrt(std::max(sum_of_squares / count - mean * mean, 0.0));
    require(deviation > 0.01, "the generator should not have collapsed to one point");
}

void test_seed_reproducibility() {
    const ml_scratch::LabeledDataset real{
        {{0.9, 0.1, 0.8, 0.2}, 0}, {{0.8, 0.2, 0.9, 0.1}, 0}, {{0.1, 0.9, 0.2, 0.8}, 1},
        {{0.2, 0.8, 0.1, 0.9}, 1}, {{0.7, 0.3, 0.7, 0.3}, 0}, {{0.3, 0.7, 0.3, 0.7}, 1}};
    ml_scratch::GanTrainingConfig config;
    config.epochs = 5;
    config.batch_size = 4;
    config.seed = 9;

    ml_scratch::ConditionalGan first{noise_size, class_count, small_generator(),
                                     small_discriminator(), 31};
    ml_scratch::ConditionalGan second{noise_size, class_count, small_generator(),
                                      small_discriminator(), 31};
    require(first.generator().parameters() == second.generator().parameters() &&
                first.discriminator().parameters() == second.discriminator().parameters(),
            "the same seed should give the same initial networks");
    // The two networks must not start as copies of each other's first layer.
    require(first.generator().parameters() != first.discriminator().parameters(),
            "the generator and discriminator should be seeded differently");

    const auto first_result = first.fit(real, config);
    const auto second_result = second.fit(real, config);
    require(first_result == second_result, "the same seed should give the same history");
    require(first.generator().parameters() == second.generator().parameters() &&
                first.discriminator().parameters() == second.discriminator().parameters(),
            "the same seed should give the same trained networks");

    ml_scratch::ConditionalGan other{noise_size, class_count, small_generator(),
                                     small_discriminator(), 32};
    require(other.generator().parameters() != first.generator().parameters(),
            "a different seed should give different networks");
}

void test_checkpoint_round_trip() {
    ml_scratch::ConditionalGan gan{noise_size, class_count, small_generator(),
                                   small_discriminator(), 31};
    const ml_scratch::LabeledDataset real{
        {{0.9, 0.1, 0.8, 0.2}, 0}, {{0.1, 0.9, 0.2, 0.8}, 1}, {{0.8, 0.2, 0.9, 0.1}, 0}};
    ml_scratch::GanTrainingConfig config;
    config.epochs = 2;
    config.batch_size = 3;
    static_cast<void>(gan.fit(real, config));

    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_gan_test.checkpoint").string();
    gan.save(path);
    const auto restored = ml_scratch::ConditionalGan::load(path);
    require(restored.noise_size() == gan.noise_size() &&
                restored.class_count() == gan.class_count() &&
                restored.sample_size() == gan.sample_size(),
            "the restored model should have the same sizes");
    require(restored.generator().parameters() == gan.generator().parameters() &&
                restored.discriminator().parameters() == gan.discriminator().parameters(),
            "the restored model should have exactly the same parameters");
    const auto noise = some_noise(1, 5).front();
    require(restored.generate(noise, 1) == gan.generate(noise, 1),
            "the restored generator should produce the same sample");
    require(restored.discriminate(real[0].features, 0) == gan.discriminate(real[0].features, 0),
            "the restored discriminator should give the same verdict");

    // A checkpoint whose networks do not fit together is rejected on load.
    const std::string broken =
        (std::filesystem::temp_directory_path() / "ml_scratch_gan_broken.checkpoint").string();
    {
        std::ofstream stream{broken};
        stream << "ml_scratch_conditional_gan 1\nnoise " << noise_size + 1 << "\nclasses "
               << class_count << '\n';
        gan.generator().save(stream, broken);
        gan.discriminator().save(stream, broken);
    }
    bool rejected = false;
    try {
        static_cast<void>(ml_scratch::ConditionalGan::load(broken));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "a checkpoint whose networks do not fit together should be rejected");
    require_runtime_error([&] { static_cast<void>(ml_scratch::ConditionalGan::load(path + ".x")); },
                          "a missing checkpoint should be reported");
    std::filesystem::remove(path);
    std::filesystem::remove(broken);
}

} // namespace

int main() {
    try {
        test_leaky_rectified_linear();
        test_input_gradient_matches_finite_differences();
        test_backpropagate_takes_an_external_output_gradient();
        test_stream_checkpoints_share_a_stream();
        test_construction_validation();
        test_generation_is_deterministic_and_conditioned();
        test_generator_gradient_matches_finite_differences();
        test_discriminator_gradient_matches_finite_differences();
        test_minimax_loss_saturates_where_the_other_does_not();
        test_training_learns_a_conditional_distribution();
        test_seed_reproducibility();
        test_checkpoint_round_trip();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
