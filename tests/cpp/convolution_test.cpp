#include "ml_scratch/convolution.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
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

// Small 1x5x5 images in two classes: a vertical bar and a horizontal bar.
ml_scratch::LabeledDataset bars_dataset() {
    const auto image = [](const std::vector<double>& pixels) { return pixels; };
    return {
        {image({0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}), 0},
        {image({0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0}), 0},
        {image({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), 1},
        {image({0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), 1},
    };
}

// Central differences over the whole dataset, used to verify the convolution gradients.
ml_scratch::GradientCheckResult check(ml_scratch::ConvolutionalNetwork& network,
                                      const ml_scratch::LabeledDataset& dataset,
                                      const double epsilon = 1e-5, const double tolerance = 1e-7) {
    const std::vector<double> analytic = network.gradient(dataset);
    const std::vector<double> original = network.parameters();
    std::vector<double> perturbed = original;
    std::vector<double> numerical(original.size(), 0.0);

    for (std::size_t index = 0; index < original.size(); ++index) {
        perturbed[index] = original[index] + epsilon;
        network.set_parameters(perturbed);
        const double raised = network.loss(dataset);
        perturbed[index] = original[index] - epsilon;
        network.set_parameters(perturbed);
        const double lowered = network.loss(dataset);
        perturbed[index] = original[index];
        numerical[index] = (raised - lowered) / (2.0 * epsilon);
    }
    network.set_parameters(original);
    return ml_scratch::check_gradient(analytic, numerical, tolerance);
}

void test_shapes() {
    // 28x28 -> conv 3x3 -> 26x26 -> pool 2 -> 13x13 -> conv 3x3 -> 11x11 -> pool 2 -> 5x5.
    ml_scratch::ConvolutionalNetwork network{
        {1, 28, 28},
        {ml_scratch::convolution(8, 3, ml_scratch::Activation::rectified_linear),
         ml_scratch::max_pooling(2),
         ml_scratch::convolution(16, 3, ml_scratch::Activation::rectified_linear),
         ml_scratch::max_pooling(2), ml_scratch::flatten(),
         ml_scratch::dense(10, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        1};

    const auto& shapes = network.shapes();
    require(shapes[1] == ml_scratch::TensorShape{8, 26, 26}, "wrong shape after the first kernel");
    require(shapes[2] == ml_scratch::TensorShape{8, 13, 13}, "wrong shape after the first pool");
    require(shapes[3] == ml_scratch::TensorShape{16, 11, 11},
            "wrong shape after the second kernel");
    require(shapes[4] == ml_scratch::TensorShape{16, 5, 5}, "wrong shape after the second pool");
    require(shapes[5] == ml_scratch::TensorShape{1, 1, 400}, "flatten produced the wrong length");
    require(network.output_shape() == ml_scratch::TensorShape{1, 1, 10}, "wrong output shape");

    // 8 * (1*9) + 8 kernels, 16 * (8*9) + 16, then 400 * 10 + 10.
    require(network.parameter_count() == 80 + 1168 + 4010, "wrong parameter count");

    // Padding restores the input size, which is what "same" convolution means.
    ml_scratch::ConvolutionalNetwork padded{
        {1, 28, 28},
        {ml_scratch::convolution(4, 3, ml_scratch::Activation::identity, 1, 1),
         ml_scratch::flatten(), ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        1};
    require(padded.shapes()[1] == ml_scratch::TensorShape{4, 28, 28},
            "padding did not preserve the spatial size");

    // A stride halves the output extent.
    ml_scratch::ConvolutionalNetwork strided{
        {1, 8, 8},
        {ml_scratch::convolution(2, 2, ml_scratch::Activation::identity, 2, 0),
         ml_scratch::flatten(), ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        1};
    require(strided.shapes()[1] == ml_scratch::TensorShape{2, 4, 4}, "wrong strided output shape");
}

void test_convolution_by_hand() {
    // One 2x2 kernel over a 3x3 input gives a 2x2 output; every value is checkable by hand.
    ml_scratch::ConvolutionalNetwork network{
        {1, 3, 3},
        {ml_scratch::convolution(1, 2, ml_scratch::Activation::identity), ml_scratch::flatten(),
         ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        1};

    // Kernel [[1, 0], [0, -1]], bias 0.5; then an identity-ish dense layer we ignore.
    std::vector<double> parameters = network.parameters();
    parameters[0] = 1.0;
    parameters[1] = 0.0;
    parameters[2] = 0.0;
    parameters[3] = -1.0;
    parameters[4] = 0.5;
    network.set_parameters(parameters);

    // Input rows 1 2 3 / 4 5 6 / 7 8 9. Window at (0,0) is 1 - 5 + 0.5 = -3.5, and so on.
    ml_scratch::ConvolutionalNetwork probe{
        {1, 3, 3},
        {ml_scratch::convolution(1, 2, ml_scratch::Activation::identity), ml_scratch::flatten(),
         ml_scratch::dense(4, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        1};
    std::vector<double> probe_parameters = probe.parameters();
    probe_parameters[0] = 1.0;
    probe_parameters[1] = 0.0;
    probe_parameters[2] = 0.0;
    probe_parameters[3] = -1.0;
    probe_parameters[4] = 0.5;
    // Make the dense layer the identity so the convolution's output is observable directly.
    for (std::size_t unit = 0; unit < 4; ++unit) {
        for (std::size_t input = 0; input < 4; ++input) {
            probe_parameters[5 + unit * 4 + input] = unit == input ? 1.0 : 0.0;
        }
        probe_parameters[5 + 16 + unit] = 0.0;
    }
    probe.set_parameters(probe_parameters);

    const auto output = probe.forward({1, 2, 3, 4, 5, 6, 7, 8, 9});
    require_near(output[0], 1.0 - 5.0 + 0.5, 1e-12, "incorrect convolution at (0, 0)");
    require_near(output[1], 2.0 - 6.0 + 0.5, 1e-12, "incorrect convolution at (0, 1)");
    require_near(output[2], 4.0 - 8.0 + 0.5, 1e-12, "incorrect convolution at (1, 0)");
    require_near(output[3], 5.0 - 9.0 + 0.5, 1e-12, "incorrect convolution at (1, 1)");
}

void test_pooling_by_hand() {
    const auto identity_dense = [](ml_scratch::ConvolutionalNetwork& network,
                                   const std::size_t units, const std::size_t offset) {
        std::vector<double> parameters = network.parameters();
        for (std::size_t unit = 0; unit < units; ++unit) {
            for (std::size_t input = 0; input < units; ++input) {
                parameters[offset + unit * units + input] = unit == input ? 1.0 : 0.0;
            }
            parameters[offset + units * units + unit] = 0.0;
        }
        network.set_parameters(parameters);
    };

    ml_scratch::ConvolutionalNetwork maximum{
        {1, 4, 4},
        {ml_scratch::max_pooling(2), ml_scratch::flatten(),
         ml_scratch::dense(4, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        1};
    identity_dense(maximum, 4, 0);

    ml_scratch::ConvolutionalNetwork mean{{1, 4, 4},
                                          {ml_scratch::average_pooling(2), ml_scratch::flatten(),
                                           ml_scratch::dense(4, ml_scratch::Activation::identity)},
                                          ml_scratch::Loss::softmax_cross_entropy,
                                          1};
    identity_dense(mean, 4, 0);

    const std::vector<double> image{1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16};
    const auto pooled = maximum.forward(image);
    require_near(pooled[0], 4.0, 1e-12, "incorrect max pool");
    require_near(pooled[1], 8.0, 1e-12, "incorrect max pool");
    require_near(pooled[2], 12.0, 1e-12, "incorrect max pool");
    require_near(pooled[3], 16.0, 1e-12, "incorrect max pool");

    const auto averaged = mean.forward(image);
    require_near(averaged[0], (1.0 + 2.0 + 3.0 + 4.0) / 4.0, 1e-12, "incorrect average pool");
    require_near(averaged[3], (13.0 + 14.0 + 15.0 + 16.0) / 4.0, 1e-12, "incorrect average pool");
}

// The real verification: every layer kind's backward pass against central differences.
void test_gradients() {
    const auto dataset = bars_dataset();

    struct Configuration {
        std::string_view name;
        std::vector<ml_scratch::ConvLayerSpec> layers;
    };
    const std::vector<Configuration> configurations{
        {"convolution then dense",
         {ml_scratch::convolution(3, 3, ml_scratch::Activation::hyperbolic_tangent),
          ml_scratch::flatten(), ml_scratch::dense(2, ml_scratch::Activation::identity)}},
        {"max pooling",
         {ml_scratch::convolution(3, 2, ml_scratch::Activation::hyperbolic_tangent),
          ml_scratch::max_pooling(2), ml_scratch::flatten(),
          ml_scratch::dense(2, ml_scratch::Activation::identity)}},
        {"average pooling",
         {ml_scratch::convolution(3, 2, ml_scratch::Activation::hyperbolic_tangent),
          ml_scratch::average_pooling(2), ml_scratch::flatten(),
          ml_scratch::dense(2, ml_scratch::Activation::identity)}},
        {"stride and padding",
         {ml_scratch::convolution(4, 3, ml_scratch::Activation::hyperbolic_tangent, 2, 1),
          ml_scratch::flatten(), ml_scratch::dense(2, ml_scratch::Activation::identity)}},
        {"two convolutions",
         {ml_scratch::convolution(3, 3, ml_scratch::Activation::hyperbolic_tangent),
          ml_scratch::convolution(2, 2, ml_scratch::Activation::hyperbolic_tangent),
          ml_scratch::flatten(), ml_scratch::dense(2, ml_scratch::Activation::identity)}},
        {"hidden dense layer",
         {ml_scratch::convolution(2, 3, ml_scratch::Activation::hyperbolic_tangent),
          ml_scratch::max_pooling(2), ml_scratch::flatten(),
          ml_scratch::dense(6, ml_scratch::Activation::hyperbolic_tangent),
          ml_scratch::dense(2, ml_scratch::Activation::identity)}},
    };

    for (const Configuration& configuration : configurations) {
        ml_scratch::ConvolutionalNetwork network{
            {1, 5, 5}, configuration.layers, ml_scratch::Loss::softmax_cross_entropy, 4242};
        const auto result = check(network, dataset);
        require(result.passed,
                std::string{"gradient check failed for "} + std::string{configuration.name});
    }

    // ReLU is checked separately and away from its kink, since a pre-activation within epsilon of
    // zero makes the difference quotient straddle a slope discontinuity.
    ml_scratch::ConvolutionalNetwork relu{
        {1, 5, 5},
        {ml_scratch::convolution(3, 3, ml_scratch::Activation::rectified_linear),
         ml_scratch::max_pooling(2), ml_scratch::flatten(),
         ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        7};
    require(check(relu, dataset, 1e-5, 1e-5).passed, "the ReLU gradient check failed");
}

void test_training_learns_bars() {
    const auto dataset = bars_dataset();
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.1;
    config.max_epochs = 500;
    config.target_loss = 0.02;
    config.optimizer.kind = ml_scratch::OptimizerKind::adam;
    config.learning_rate = 0.01;

    ml_scratch::ConvolutionalNetwork network{
        {1, 5, 5},
        {ml_scratch::convolution(4, 3, ml_scratch::Activation::rectified_linear),
         ml_scratch::max_pooling(2), ml_scratch::flatten(),
         ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        11};
    const auto result = network.fit(dataset, config);

    require(result.converged, "the convolutional network did not learn the bars");
    require_near(network.accuracy(dataset), 1.0, 1e-12, "a bar was misclassified");

    const auto confusion = network.confusion_matrix(dataset);
    require(confusion.size() == 2, "wrong confusion matrix size");
    require(confusion[0][0] == 2 && confusion[1][1] == 2, "incorrect confusion matrix");

    // Probabilities must form a distribution.
    const auto probabilities = network.predict(dataset.front().features);
    double total = 0.0;
    for (const double probability : probabilities) {
        require(probability >= 0.0 && probability <= 1.0, "a probability left [0, 1]");
        total += probability;
    }
    require_near(total, 1.0, 1e-12, "softmax outputs do not sum to one");
}

// Weight sharing is the defining property: the same kernel is applied at every position, so a
// pattern learned in one place is recognized in another.
void test_weight_sharing_gives_translation_equivariance() {
    ml_scratch::ConvolutionalNetwork network{
        {1, 5, 5},
        {ml_scratch::convolution(1, 2, ml_scratch::Activation::identity), ml_scratch::flatten(),
         ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        3};

    std::vector<double> parameters = network.parameters();
    parameters[0] = 1.0;
    parameters[1] = -1.0;
    parameters[2] = -1.0;
    parameters[3] = 1.0;
    parameters[4] = 0.0;
    network.set_parameters(parameters);

    // The convolution has 4x4 = 16 outputs. A feature placed at (0, 0) and the same feature at
    // (2, 2) must produce the same response, just moved.
    ml_scratch::ConvolutionalNetwork probe{
        {1, 5, 5},
        {ml_scratch::convolution(1, 2, ml_scratch::Activation::identity), ml_scratch::flatten(),
         ml_scratch::dense(16, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        3};
    std::vector<double> probe_parameters = probe.parameters();
    probe_parameters[0] = 1.0;
    probe_parameters[1] = -1.0;
    probe_parameters[2] = -1.0;
    probe_parameters[3] = 1.0;
    probe_parameters[4] = 0.0;
    for (std::size_t unit = 0; unit < 16; ++unit) {
        for (std::size_t input = 0; input < 16; ++input) {
            probe_parameters[5 + unit * 16 + input] = unit == input ? 1.0 : 0.0;
        }
        probe_parameters[5 + 256 + unit] = 0.0;
    }
    probe.set_parameters(probe_parameters);

    std::vector<double> corner(25, 0.0);
    corner[0] = 1.0; // a single lit pixel at (0, 0)
    std::vector<double> shifted(25, 0.0);
    shifted[2 * 5 + 2] = 1.0; // the same pixel at (2, 2)

    const auto corner_response = probe.forward(corner);
    const auto shifted_response = probe.forward(shifted);
    // Output (r, c) of the shifted image must equal output (r - 2, c - 2) of the original.
    require_near(shifted_response[2 * 4 + 2], corner_response[0], 1e-12,
                 "the same kernel gave different responses at different positions");
    require_near(shifted_response[2 * 4 + 3], corner_response[1], 1e-12,
                 "the response did not translate with the feature");
}

void test_checkpoint_round_trip() {
    const auto dataset = bars_dataset();
    ml_scratch::ConvolutionalNetwork network{
        {1, 5, 5},
        {ml_scratch::convolution(3, 3, ml_scratch::Activation::rectified_linear, 1, 1),
         ml_scratch::max_pooling(2), ml_scratch::flatten(),
         ml_scratch::dense(4, ml_scratch::Activation::hyperbolic_tangent),
         ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        19};

    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.05;
    config.max_epochs = 40;
    static_cast<void>(network.fit(dataset, config));

    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_conv.checkpoint").string();
    network.save(path);
    const auto restored = ml_scratch::ConvolutionalNetwork::load(path);

    require(restored.layers() == network.layers(), "the architecture did not survive");
    require(restored.shapes() == network.shapes(), "the shapes did not survive");
    require(restored.parameters() == network.parameters(),
            "parameters did not survive the round trip exactly");
    require_near(restored.loss(dataset), network.loss(dataset), 0.0,
                 "a restored network scored differently");

    // Corruption surfaces through two different paths, so both are checked. A truncated payload is
    // caught by the reader itself; an architecture that cannot be built is caught by the
    // constructor's own validation, which reports a logic error rather than a runtime one.
    const auto write_broken = [](const std::string& name, const std::string& text) {
        const std::string target = (std::filesystem::temp_directory_path() / name).string();
        std::ofstream stream{target};
        stream << text;
        return target;
    };
    const std::string truncated =
        write_broken("ml_scratch_conv_short.checkpoint",
                     "ml_scratch_convolutional 1\nloss 2\ninput 1 5 5\nlayers 2\n"
                     "3 0 0 0 1 0 0 0 0\n4 0 0 0 1 0 0 0 2\nparameters 52\n1\n2\n");
    const std::string impossible =
        write_broken("ml_scratch_conv_bad.checkpoint",
                     "ml_scratch_convolutional 1\nloss 2\ninput 1 5 5\nlayers 1\n"
                     "4 0 0 0 1 0 0 0 3\nparameters 2\n1\n");
    const std::string wrong_magic = write_broken("ml_scratch_conv_magic.checkpoint", "nope 1\n");

    for (const std::string& target : {truncated, impossible, wrong_magic}) {
        bool rejected = false;
        try {
            static_cast<void>(ml_scratch::ConvolutionalNetwork::load(target));
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "a corrupted checkpoint was accepted");
    }

    std::filesystem::remove(path);
    for (const std::string& target : {truncated, impossible, wrong_magic}) {
        std::filesystem::remove(target);
    }
}

void test_input_validation() {
    require_invalid_argument(
        [] {
            ml_scratch::ConvolutionalNetwork network{
                {1, 5, 5}, {}, ml_scratch::Loss::softmax_cross_entropy};
        },
        "an empty network was accepted");
    // A 7x7 kernel cannot fit a 5x5 input.
    require_invalid_argument(
        [] {
            ml_scratch::ConvolutionalNetwork network{
                {1, 5, 5},
                {ml_scratch::convolution(2, 7, ml_scratch::Activation::identity),
                 ml_scratch::flatten(), ml_scratch::dense(2, ml_scratch::Activation::identity)},
                ml_scratch::Loss::softmax_cross_entropy};
        },
        "an oversized kernel was accepted");
    // A dense layer needs a flattened input.
    require_invalid_argument(
        [] {
            ml_scratch::ConvolutionalNetwork network{
                {1, 5, 5},
                {ml_scratch::convolution(2, 3, ml_scratch::Activation::identity),
                 ml_scratch::dense(2, ml_scratch::Activation::identity)},
                ml_scratch::Loss::softmax_cross_entropy};
        },
        "a dense layer accepted a shaped input");
    // Cross-entropy needs a linear final layer.
    require_invalid_argument(
        [] {
            ml_scratch::ConvolutionalNetwork network{
                {1, 5, 5},
                {ml_scratch::flatten(), ml_scratch::dense(2, ml_scratch::Activation::sigmoid)},
                ml_scratch::Loss::softmax_cross_entropy};
        },
        "a squashed output under a logit loss was accepted");

    ml_scratch::ConvolutionalNetwork network{
        {1, 5, 5},
        {ml_scratch::convolution(2, 3, ml_scratch::Activation::identity), ml_scratch::flatten(),
         ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy};
    require_invalid_argument([&] { static_cast<void>(network.forward({1.0, 2.0})); },
                             "a wrong input size was accepted");
    require_invalid_argument([&] { network.set_parameters({1.0}); },
                             "a wrong parameter vector size was accepted");
    require_invalid_argument([&] { static_cast<void>(network.accuracy({{{1.0}, 0}})); },
                             "a mismatched dataset was accepted");

    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.0;
    require_invalid_argument([&] { network.fit(bars_dataset(), config); },
                             "a zero learning rate was accepted");
}

} // namespace

int main() {
    try {
        test_shapes();
        test_convolution_by_hand();
        test_pooling_by_hand();
        test_gradients();
        test_training_learns_bars();
        test_weight_sharing_gives_translation_equivariance();
        test_checkpoint_round_trip();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
