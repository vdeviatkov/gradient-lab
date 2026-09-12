#include "ml_scratch/neural_network.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

const ml_scratch::SupervisedDataset xor_data{
    {{0.0, 0.0}, {0.0}},
    {{0.0, 1.0}, {1.0}},
    {{1.0, 0.0}, {1.0}},
    {{1.0, 1.0}, {0.0}},
};

const ml_scratch::SupervisedDataset three_class_data{
    {{-1.0, -1.0}, {1.0, 0.0, 0.0}}, {{-0.8, -1.2}, {1.0, 0.0, 0.0}},
    {{1.0, -1.0}, {0.0, 1.0, 0.0}},  {{1.2, -0.7}, {0.0, 1.0, 0.0}},
    {{0.0, 1.5}, {0.0, 0.0, 1.0}},   {{-0.3, 1.1}, {0.0, 0.0, 1.0}},
};

const ml_scratch::SupervisedDataset regression_data{
    {{0.3, -0.7}, {0.4, -0.2}},
    {{-0.5, 0.9}, {-0.1, 0.8}},
    {{0.8, 0.2}, {0.6, 0.1}},
};

void test_forward_pass_by_hand() {
    // A single linear unit with known parameters must reproduce w.x + b exactly.
    ml_scratch::FeedForwardNetwork network{{{2, 1, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::mean_squared_error};
    require(network.parameter_count() == 3, "a 2 -> 1 linear layer has three parameters");
    network.set_parameters({2.0, -3.0, 0.5});
    require_near(network.forward({1.0, 1.0})[0], -0.5, 1e-12, "incorrect linear forward pass");

    // Adding a tanh hidden layer: parameters are laid out layer by layer, weights then biases.
    ml_scratch::FeedForwardNetwork deep{{{1, 2, ml_scratch::Activation::hyperbolic_tangent},
                                         {2, 1, ml_scratch::Activation::identity}},
                                        ml_scratch::Loss::mean_squared_error};
    require(deep.parameter_count() == 2 + 2 + 2 + 1, "incorrect parameter count");
    deep.set_parameters({1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.25});
    const double hidden = std::tanh(1.0);
    require_near(deep.forward({1.0})[0], hidden - hidden + 0.25, 1e-12,
                 "incorrect two-layer forward pass");

    // Round-tripping the flat parameter vector must be lossless.
    const auto values = deep.parameters();
    deep.set_parameters(values);
    require(deep.parameters() == values, "parameter round trip changed the network");
}

void test_loss_values_by_hand() {
    ml_scratch::FeedForwardNetwork mse{{{1, 1, ml_scratch::Activation::identity}},
                                       ml_scratch::Loss::mean_squared_error};
    mse.set_parameters({0.0, 0.0});
    require_near(mse.loss(ml_scratch::SupervisedDataset{{{1.0}, {2.0}}}), 4.0, 1e-12,
                 "incorrect squared error");

    // At zero logits, sigmoid and softmax are uniform, so both losses take their known values.
    ml_scratch::FeedForwardNetwork bce{{{1, 1, ml_scratch::Activation::identity}},
                                       ml_scratch::Loss::binary_cross_entropy};
    bce.set_parameters({0.0, 0.0});
    require_near(bce.loss(ml_scratch::SupervisedDataset{{{1.0}, {1.0}}}), std::log(2.0), 1e-12,
                 "zero-logit binary cross-entropy should equal log(2)");

    ml_scratch::FeedForwardNetwork softmax{{{1, 3, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::softmax_cross_entropy};
    softmax.set_parameters(std::vector<double>(softmax.parameter_count(), 0.0));
    require_near(softmax.loss(ml_scratch::SupervisedDataset{{{1.0}, {1.0, 0.0, 0.0}}}),
                 std::log(3.0), 1e-12, "zero-logit softmax cross-entropy should equal log(3)");

    // Saturated logits must not overflow into a non-finite loss.
    ml_scratch::FeedForwardNetwork saturated{{{1, 1, ml_scratch::Activation::identity}},
                                             ml_scratch::Loss::binary_cross_entropy};
    saturated.set_parameters({1000.0, 0.0});
    const double loss = saturated.loss(ml_scratch::SupervisedDataset{{{1.0}, {0.0}}});
    require(std::isfinite(loss) && loss > 0.0, "binary cross-entropy overflowed at a large logit");
    require_near(saturated.predict({1.0})[0], 1.0, 1e-12, "saturated sigmoid was not stable");
}

void test_analytic_gradient_by_hand() {
    // For one linear unit under squared error, dL/dw = 2(z - y)x and dL/db = 2(z - y).
    ml_scratch::FeedForwardNetwork network{{{2, 1, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::mean_squared_error};
    network.set_parameters({0.0, 0.0, 0.0});
    const auto gradient = network.gradient(ml_scratch::SupervisedDataset{{{3.0, -1.0}, {2.0}}});
    require_near(gradient[0], -12.0, 1e-12, "incorrect weight gradient");
    require_near(gradient[1], 4.0, 1e-12, "incorrect weight gradient");
    require_near(gradient[2], -4.0, 1e-12, "incorrect bias gradient");

    // With sigmoid folded into binary cross-entropy the output delta collapses to p - y.
    ml_scratch::FeedForwardNetwork logistic{{{1, 1, ml_scratch::Activation::identity}},
                                            ml_scratch::Loss::binary_cross_entropy};
    logistic.set_parameters({0.0, 0.0});
    const auto logistic_gradient = logistic.gradient(ml_scratch::SupervisedDataset{{{2.0}, {1.0}}});
    require_near(logistic_gradient[0], -1.0, 1e-12, "incorrect logistic weight gradient");
    require_near(logistic_gradient[1], -0.5, 1e-12, "incorrect logistic bias gradient");
}

// The core of this milestone: every architecture and loss must agree with central differences.
void test_gradient_check_across_configurations() {
    struct Configuration {
        std::string_view name;
        std::vector<ml_scratch::DenseLayer> layers;
        ml_scratch::Loss loss;
        const ml_scratch::SupervisedDataset* data;
    };

    const std::vector<Configuration> configurations{
        {"linear regression",
         {{2, 2, ml_scratch::Activation::identity}},
         ml_scratch::Loss::mean_squared_error,
         &regression_data},
        {"tanh hidden, squared error",
         {{2, 5, ml_scratch::Activation::hyperbolic_tangent},
          {5, 2, ml_scratch::Activation::identity}},
         ml_scratch::Loss::mean_squared_error,
         &regression_data},
        {"sigmoid output under squared error",
         {{2, 4, ml_scratch::Activation::hyperbolic_tangent},
          {4, 2, ml_scratch::Activation::sigmoid}},
         ml_scratch::Loss::mean_squared_error,
         &regression_data},
        {"binary cross-entropy",
         {{2, 4, ml_scratch::Activation::hyperbolic_tangent},
          {4, 1, ml_scratch::Activation::identity}},
         ml_scratch::Loss::binary_cross_entropy,
         &xor_data},
        {"softmax cross-entropy",
         {{2, 6, ml_scratch::Activation::hyperbolic_tangent},
          {6, 3, ml_scratch::Activation::identity}},
         ml_scratch::Loss::softmax_cross_entropy,
         &three_class_data},
        {"three hidden layers, mixed activations",
         {{2, 4, ml_scratch::Activation::hyperbolic_tangent},
          {4, 4, ml_scratch::Activation::sigmoid},
          {4, 3, ml_scratch::Activation::hyperbolic_tangent},
          {3, 3, ml_scratch::Activation::identity}},
         ml_scratch::Loss::softmax_cross_entropy,
         &three_class_data},
    };

    for (const Configuration& configuration : configurations) {
        ml_scratch::FeedForwardNetwork network{configuration.layers, configuration.loss, 4242};
        const auto result = ml_scratch::check_gradient(network, *configuration.data);
        require(result.passed,
                std::string{"gradient check failed for "} + std::string{configuration.name});
    }
}

// A checker that never fails is worthless, so corrupt one parameter's gradient and require a catch.
void test_gradient_check_detects_a_wrong_gradient() {
    ml_scratch::FeedForwardNetwork network{{{2, 4, ml_scratch::Activation::hyperbolic_tangent},
                                            {4, 1, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::binary_cross_entropy,
                                           7};
    auto analytic = network.gradient(xor_data);
    const auto numerical = network.numerical_gradient(xor_data);
    require(ml_scratch::check_gradient(analytic, numerical).passed,
            "the correct gradient did not pass");

    // A sign flip is the classic backpropagation bug.
    analytic[3] = -analytic[3];
    const auto flipped = ml_scratch::check_gradient(analytic, numerical);
    require(!flipped.passed, "a flipped gradient sign was not detected");
    require(flipped.worst_parameter == 3, "the wrong parameter was blamed");

    // So is dropping the activation derivative, which scales one entry.
    analytic[3] = -analytic[3];
    analytic[0] *= 1.01;
    const auto scaled = ml_scratch::check_gradient(analytic, numerical);
    require(!scaled.passed, "a one percent gradient error was not detected");
    require(scaled.worst_parameter == 0, "the wrong parameter was blamed");

    // Identical gradients agree perfectly, including the all-zero pair.
    const auto zeros = ml_scratch::check_gradient({0.0, 0.0}, {0.0, 0.0});
    require(zeros.passed && zeros.relative_error == 0.0, "identical zero gradients did not pass");

    // When the whole gradient vanishes there is nothing to normalize against, so the norm ratio
    // reports disagreement even though the two gradients are the same to within 1e-18. This is
    // the degeneracy the experiment measures at convergence: the absolute figure is what still
    // carries information once every entry approaches zero.
    const auto vanishing = ml_scratch::check_gradient({0.0, 1e-18}, {0.0, -1e-18});
    require(!vanishing.passed, "a vanishing gradient pair passed the norm ratio");
    require(vanishing.max_absolute_error <= 1e-17,
            "absolute disagreement should still be negligible");
}

void test_training_learns_xor() {
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.5;
    config.max_epochs = 20'000;
    config.target_loss = 0.02;

    ml_scratch::FeedForwardNetwork network{{{2, 4, ml_scratch::Activation::hyperbolic_tangent},
                                            {4, 1, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::binary_cross_entropy,
                                           20260910};
    const auto result = network.fit(xor_data, config);

    require(result.converged, "the general network did not learn XOR");
    require(result.epochs == result.loss_per_epoch.size(), "loss history has the wrong size");
    require(result.loss_per_epoch.back() <= config.target_loss,
            "reported convergence above the target loss");
    for (const auto& sample : xor_data) {
        const double probability = network.predict(sample.features)[0];
        const double expected = sample.targets[0];
        require((probability >= 0.5) == (expected == 1.0), "XOR prediction is wrong");
    }

    // The gradient must still check out at the trained parameters, not only at initialization.
    require(ml_scratch::check_gradient(network, xor_data).passed,
            "gradient check failed at the trained parameters");
}

void test_training_learns_three_classes() {
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.2;
    config.max_epochs = 5'000;
    config.target_loss = 0.05;

    ml_scratch::FeedForwardNetwork network{{{2, 6, ml_scratch::Activation::hyperbolic_tangent},
                                            {6, 3, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::softmax_cross_entropy,
                                           11};
    const auto result = network.fit(three_class_data, config);
    require(result.converged, "the softmax network did not converge");
    require_near(network.accuracy(three_class_data), 1.0, 1e-12,
                 "the softmax network misclassified a training sample");

    // Predicted probabilities must form a distribution.
    const auto probabilities = network.predict({0.0, 1.5});
    double total = 0.0;
    for (const double probability : probabilities) {
        require(probability >= 0.0 && probability <= 1.0, "a probability left [0, 1]");
        total += probability;
    }
    require_near(total, 1.0, 1e-12, "softmax outputs do not sum to one");
}

// The class-index overloads must compute exactly what the one-hot ones do, since the experiment
// relies on them for the whole MNIST training set.
void test_labeled_dataset_path_matches_one_hot() {
    const ml_scratch::LabeledDataset labeled{
        {{-1.0, -1.0}, 0}, {{-0.8, -1.2}, 0}, {{1.0, -1.0}, 1},
        {{1.2, -0.7}, 1},  {{0.0, 1.5}, 2},   {{-0.3, 1.1}, 2},
    };
    const auto encoded = ml_scratch::to_one_hot(labeled, 3);

    const std::vector<ml_scratch::DenseLayer> layers{
        {2, 5, ml_scratch::Activation::hyperbolic_tangent},
        {5, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::softmax_cross_entropy, 31};

    require_near(network.loss(labeled), network.loss(encoded), 1e-15,
                 "the two dataset representations disagree on the loss");
    require_near(network.accuracy(labeled), network.accuracy(encoded), 0.0,
                 "the two dataset representations disagree on accuracy");

    const auto labeled_gradient = network.gradient(labeled);
    const auto encoded_gradient = network.gradient(encoded);
    for (std::size_t index = 0; index < labeled_gradient.size(); ++index) {
        require_near(labeled_gradient[index], encoded_gradient[index], 1e-15,
                     "the two dataset representations disagree on the gradient");
    }

    // Training through either path must follow the identical trajectory.
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.3;
    config.max_epochs = 25;
    config.batch_size = 2;
    config.seed = 8;

    ml_scratch::FeedForwardNetwork from_labels{layers, ml_scratch::Loss::softmax_cross_entropy, 31};
    ml_scratch::FeedForwardNetwork from_one_hot{layers, ml_scratch::Loss::softmax_cross_entropy,
                                                31};
    const auto labeled_result = from_labels.fit(labeled, config);
    const auto encoded_result = from_one_hot.fit(encoded, config);
    require(labeled_result.epochs == encoded_result.epochs, "training paths diverged");
    const auto labeled_parameters = from_labels.parameters();
    const auto encoded_parameters = from_one_hot.parameters();
    for (std::size_t index = 0; index < labeled_parameters.size(); ++index) {
        require_near(labeled_parameters[index], encoded_parameters[index], 1e-12,
                     "training paths produced different parameters");
    }

    // The overloads are classification-only.
    ml_scratch::FeedForwardNetwork regressor{{{2, 1, ml_scratch::Activation::identity}},
                                             ml_scratch::Loss::mean_squared_error};
    require_invalid_argument([&] { static_cast<void>(regressor.loss(labeled)); },
                             "a non-softmax loss accepted class indices");
}

void test_confusion_matrix() {
    const ml_scratch::LabeledDataset data{
        {{-1.0, -1.0}, 0}, {{-0.8, -1.2}, 0}, {{1.0, -1.0}, 1},
        {{1.2, -0.7}, 1},  {{0.0, 1.5}, 2},   {{-0.3, 1.1}, 2},
    };
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.3;
    config.max_epochs = 2'000;
    config.target_loss = 0.05;

    ml_scratch::FeedForwardNetwork network{{{2, 6, ml_scratch::Activation::hyperbolic_tangent},
                                            {6, 3, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::softmax_cross_entropy,
                                           5};
    static_cast<void>(network.fit(data, config));

    const auto confusion = network.confusion_matrix(data);
    require(confusion.size() == 3, "confusion matrix has the wrong shape");
    std::size_t diagonal = 0;
    std::size_t total = 0;
    for (std::size_t actual = 0; actual < 3; ++actual) {
        diagonal += confusion[actual][actual];
        for (const std::size_t count : confusion[actual]) {
            total += count;
        }
    }
    require(total == data.size(), "confusion matrix lost samples");
    require_near(static_cast<double>(diagonal) / static_cast<double>(total), network.accuracy(data),
                 1e-12, "confusion diagonal disagrees with accuracy");
}

// Training an MNIST-sized network takes minutes, so its parameters must survive a round trip.
void test_checkpoint_round_trip() {
    const std::vector<ml_scratch::DenseLayer> layers{
        {2, 4, ml_scratch::Activation::hyperbolic_tangent},
        {4, 3, ml_scratch::Activation::sigmoid},
        {3, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::softmax_cross_entropy, 77};
    const ml_scratch::LabeledDataset data{
        {{-1.0, -1.0}, 0},
        {{1.0, -1.0}, 1},
        {{0.0, 1.5}, 2},
    };
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.2;
    config.max_epochs = 50;
    static_cast<void>(network.fit(data, config));

    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_network.checkpoint").string();
    network.save(path);
    const auto restored = ml_scratch::FeedForwardNetwork::load(path);

    require(restored.layers() == network.layers(),
            "the architecture did not survive the round trip");
    require(restored.loss_function() == network.loss_function(), "the loss did not survive");
    require(restored.parameter_count() == network.parameter_count(), "parameter count changed");
    // 17 significant digits must reproduce every double exactly, not approximately.
    require(restored.parameters() == network.parameters(),
            "parameters did not survive the round trip exactly");
    for (const auto& sample : data) {
        require(restored.predict_class(sample.features) == network.predict_class(sample.features),
                "a restored network predicted differently");
    }
    require_near(restored.loss(data), network.loss(data), 0.0,
                 "a restored network scored differently");

    // Corrupted checkpoints must be rejected rather than silently loaded.
    const auto write_text = [](const std::string& name, const std::string& text) {
        const std::string target = (std::filesystem::temp_directory_path() / name).string();
        std::ofstream stream{target};
        stream << text;
        return target;
    };
    const std::string wrong_magic = write_text("ml_scratch_bad_magic.checkpoint", "something 1\n");
    const std::string wrong_count =
        write_text("ml_scratch_bad_count.checkpoint",
                   "ml_scratch_feedforward 1\nloss 2\nlayers 1\n2 3 0\nparameters 4\n1\n2\n3\n4\n");
    const std::string truncated =
        write_text("ml_scratch_truncated.checkpoint",
                   "ml_scratch_feedforward 1\nloss 2\nlayers 1\n2 3 0\nparameters 9\n1\n2\n");
    const std::string bad_activation =
        write_text("ml_scratch_bad_activation.checkpoint",
                   "ml_scratch_feedforward 1\nloss 2\nlayers 1\n2 3 99\nparameters 9\n");

    for (const std::string& broken : {wrong_magic, wrong_count, truncated, bad_activation}) {
        bool rejected = false;
        try {
            static_cast<void>(ml_scratch::FeedForwardNetwork::load(broken));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "a corrupted checkpoint was accepted");
    }
    bool missing_rejected = false;
    try {
        static_cast<void>(ml_scratch::FeedForwardNetwork::load("/no/such/checkpoint"));
    } catch (const std::runtime_error&) {
        missing_rejected = true;
    }
    require(missing_rejected, "a missing checkpoint was accepted");

    for (const std::string& target : {path, wrong_magic, wrong_count, truncated, bad_activation}) {
        std::filesystem::remove(target);
    }
}

void test_seed_reproducibility() {
    const std::vector<ml_scratch::DenseLayer> layers{
        {2, 4, ml_scratch::Activation::hyperbolic_tangent},
        {4, 1, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork first{layers, ml_scratch::Loss::binary_cross_entropy, 99};
    ml_scratch::FeedForwardNetwork second{layers, ml_scratch::Loss::binary_cross_entropy, 99};
    require(first.parameters() == second.parameters(), "the same seed initialized differently");

    ml_scratch::FeedForwardNetwork other{layers, ml_scratch::Loss::binary_cross_entropy, 100};
    require(first.parameters() != other.parameters(), "different seeds initialized identically");

    ml_scratch::NetworkTrainingConfig config;
    config.max_epochs = 50;
    config.batch_size = 2;
    config.seed = 5;
    const auto first_result = first.fit(xor_data, config);
    const auto second_result = second.fit(xor_data, config);
    require(first_result == second_result, "identical seeds produced different histories");
    require(first.parameters() == second.parameters(),
            "identical seeds produced different weights");
}

void test_input_validation() {
    require_invalid_argument(
        [] { ml_scratch::FeedForwardNetwork network{{}, ml_scratch::Loss::mean_squared_error}; },
        "an empty network was accepted");
    require_invalid_argument(
        [] {
            ml_scratch::FeedForwardNetwork network{{{2, 3, ml_scratch::Activation::identity},
                                                    {4, 1, ml_scratch::Activation::identity}},
                                                   ml_scratch::Loss::mean_squared_error};
        },
        "disconnected layer sizes were accepted");
    require_invalid_argument(
        [] {
            ml_scratch::FeedForwardNetwork network{{{2, 1, ml_scratch::Activation::sigmoid}},
                                                   ml_scratch::Loss::binary_cross_entropy};
        },
        "a squashed output under a logit loss was accepted");
    require_invalid_argument(
        [] {
            ml_scratch::FeedForwardNetwork network{{{2, 1, ml_scratch::Activation::identity}},
                                                   ml_scratch::Loss::softmax_cross_entropy};
        },
        "a single-output softmax was accepted");
    require_invalid_argument(
        [] {
            ml_scratch::FeedForwardNetwork network{{{0, 1, ml_scratch::Activation::identity}},
                                                   ml_scratch::Loss::mean_squared_error};
        },
        "a zero-sized layer was accepted");

    ml_scratch::FeedForwardNetwork network{{{2, 1, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::mean_squared_error};
    require_invalid_argument([&] { network.set_parameters({1.0}); },
                             "a wrong parameter vector size was accepted");
    require_invalid_argument(
        [&] { network.set_parameters({1.0, 1.0, std::numeric_limits<double>::infinity()}); },
        "a non-finite parameter was accepted");
    require_invalid_argument([&] { static_cast<void>(network.forward({1.0})); },
                             "a wrong input size was accepted");
    require_invalid_argument(
        [&] {
            static_cast<void>(
                network.loss(ml_scratch::SupervisedDataset{{{1.0, 2.0}, {1.0, 2.0}}}));
        },
        "a wrong target size was accepted");
    require_invalid_argument(
        [&] { static_cast<void>(network.numerical_gradient(regression_data, 0.0)); },
        "a zero epsilon was accepted");
    require_invalid_argument(
        [] { static_cast<void>(ml_scratch::check_gradient({1.0}, {1.0, 2.0})); },
        "mismatched gradient sizes were accepted");

    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.0;
    require_invalid_argument([&] { network.fit(regression_data, config); },
                             "a zero learning rate was accepted");
}

} // namespace

int main() {
    try {
        test_forward_pass_by_hand();
        test_loss_values_by_hand();
        test_analytic_gradient_by_hand();
        test_gradient_check_across_configurations();
        test_gradient_check_detects_a_wrong_gradient();
        test_training_learns_xor();
        test_training_learns_three_classes();
        test_labeled_dataset_path_matches_one_hot();
        test_confusion_matrix();
        test_checkpoint_round_trip();
        test_seed_reproducibility();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
