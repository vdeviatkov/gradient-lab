#include "ml_scratch/cifar10.hpp"
#include "ml_scratch/convolution.hpp"
#include "ml_scratch/neural_network.hpp"

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

const ml_scratch::LabeledDataset four_feature_data{
    {{-1.0, -1.0, 0.2, 0.0}, 0}, {{-0.8, -1.2, 0.1, 0.3}, 0}, {{1.0, -1.0, -0.2, 0.4}, 1},
    {{1.2, -0.7, 0.0, -0.1}, 1}, {{0.0, 1.5, 0.3, 0.2}, 2},   {{-0.3, 1.1, -0.4, 0.1}, 2},
};

void test_skip_is_added_before_the_activation() {
    // A residual layer with zero weights and zero bias computes g(0 + x), so with the identity
    // activation it is exactly the identity function.
    ml_scratch::FeedForwardNetwork identity{
        {{3, 3, ml_scratch::Activation::identity, ml_scratch::Normalization::none, 0.0, true}},
        ml_scratch::Loss::mean_squared_error,
        4};
    identity.set_parameters(std::vector<double>(identity.parameter_count(), 0.0));

    const std::vector<double> input{0.5, -1.25, 2.0};
    const auto output = identity.forward(input);
    for (std::size_t index = 0; index < input.size(); ++index) {
        require_near(output[index], input[index], 1e-15,
                     "a zeroed residual layer is not the identity");
    }

    // The same layer without the skip outputs zeros, which isolates the skip's contribution.
    ml_scratch::FeedForwardNetwork plain{
        {{3, 3, ml_scratch::Activation::identity}}, ml_scratch::Loss::mean_squared_error, 4};
    plain.set_parameters(std::vector<double>(plain.parameter_count(), 0.0));
    for (const double value : plain.forward(input)) {
        require_near(value, 0.0, 0.0, "a zeroed plain layer should output zeros");
    }

    // With a ReLU the skip is gated by the activation, which is what "added before the
    // activation" means: a negative input stays clipped.
    ml_scratch::FeedForwardNetwork gated{{{3, 3, ml_scratch::Activation::rectified_linear,
                                           ml_scratch::Normalization::none, 0.0, true}},
                                         ml_scratch::Loss::mean_squared_error,
                                         4};
    gated.set_parameters(std::vector<double>(gated.parameter_count(), 0.0));
    const auto clipped = gated.forward(input);
    require_near(clipped[0], 0.5, 1e-15, "a positive value should pass the skip");
    require_near(clipped[1], 0.0, 1e-15, "a negative value should be clipped after the skip");
}

void test_residual_gradients() {
    const std::vector<ml_scratch::DenseLayer> layers{
        {4, 6, ml_scratch::Activation::hyperbolic_tangent},
        {6, 6, ml_scratch::Activation::hyperbolic_tangent, ml_scratch::Normalization::none, 0.0,
         true},
        {6, 6, ml_scratch::Activation::rectified_linear, ml_scratch::Normalization::none, 0.0,
         true},
        {6, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::softmax_cross_entropy, 31};
    require(
        ml_scratch::check_gradient(network, ml_scratch::to_one_hot(four_feature_data, 3)).passed,
        "the residual dense gradient failed its check");

    // A skip must compose with normalization, since a residual block normally carries one.
    const std::vector<ml_scratch::DenseLayer> normalized{
        {4, 6, ml_scratch::Activation::hyperbolic_tangent},
        {6, 6, ml_scratch::Activation::hyperbolic_tangent, ml_scratch::Normalization::layer, 0.0,
         true},
        {6, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork with_norm{normalized, ml_scratch::Loss::softmax_cross_entropy,
                                             31};
    require(
        ml_scratch::check_gradient(with_norm, ml_scratch::to_one_hot(four_feature_data, 3)).passed,
        "the residual gradient failed alongside layer normalization");

    // The batch path must agree with the per-sample one on a residual network.
    const auto per_sample = network.gradient(four_feature_data);
    const auto batched = network.batch_training_gradient(four_feature_data);
    for (std::size_t index = 0; index < per_sample.size(); ++index) {
        require_near(per_sample[index], batched[index], 1e-14,
                     "the two paths disagree on a residual network");
    }
}

// The reason skips exist: in a deep stack the gradient reaching the first layer is a product of
// many Jacobians and decays geometrically. An identity path adds a term that is not multiplied by
// any weight matrix, so the early layers keep receiving signal.
void test_skips_preserve_early_gradients() {
    const auto first_layer_gradient_norm = [](const bool residual, const std::size_t depth) {
        std::vector<ml_scratch::DenseLayer> layers{
            {4, 12, ml_scratch::Activation::hyperbolic_tangent}};
        for (std::size_t index = 0; index < depth; ++index) {
            layers.push_back({12, 12, ml_scratch::Activation::hyperbolic_tangent,
                              ml_scratch::Normalization::none, 0.0, residual});
        }
        layers.push_back({12, 3, ml_scratch::Activation::identity});

        ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::softmax_cross_entropy, 9};
        const auto gradient = network.gradient(four_feature_data);
        // The first layer's block is 4 * 12 weights followed by 12 biases.
        double total = 0.0;
        for (std::size_t index = 0; index < 4 * 12 + 12; ++index) {
            total += gradient[index] * gradient[index];
        }
        return std::sqrt(total);
    };

    const double plain_shallow = first_layer_gradient_norm(false, 2);
    const double plain_deep = first_layer_gradient_norm(false, 40);
    const double residual_deep = first_layer_gradient_norm(true, 40);

    // Depth costs a plain stack most of its first-layer gradient. The decay is not monotone in
    // depth and not catastrophic: Glorot initialization is designed to hold the variance steady
    // through a tanh stack, so it already mitigates much of the problem. The effect is measurable
    // rather than dramatic, which is why the thresholds here are stated from the measurement.
    require(plain_deep < plain_shallow * 0.2,
            "a deep plain stack unexpectedly kept its early gradient");
    // The skip keeps the early gradient alive by more than two orders of magnitude.
    require(residual_deep > plain_deep * 50.0,
            "the identity skip did not preserve the early gradient");
}

void test_residual_convolution() {
    // A residual convolution must preserve its shape, so the filter count and the spatial size
    // both have to match its input.
    require_invalid_argument(
        [] {
            ml_scratch::ConvolutionalNetwork network{
                {4, 6, 6},
                {ml_scratch::residual_convolution(8, 3, ml_scratch::Activation::identity),
                 ml_scratch::flatten(), ml_scratch::dense(2, ml_scratch::Activation::identity)},
                ml_scratch::Loss::softmax_cross_entropy};
        },
        "a residual convolution accepted a changed channel count");
    require_invalid_argument(
        [] {
            static_cast<void>(
                ml_scratch::residual_convolution(4, 2, ml_scratch::Activation::identity));
        },
        "a residual convolution accepted an even kernel");

    // Zeroed kernels make the block the identity, exactly as for the dense case.
    ml_scratch::ConvolutionalNetwork identity{
        {2, 4, 4},
        {ml_scratch::residual_convolution(2, 3, ml_scratch::Activation::identity),
         ml_scratch::flatten(), ml_scratch::dense(32, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        6};
    std::vector<double> parameters(identity.parameter_count(), 0.0);
    // Make the dense layer the identity so the block's output is observable.
    const std::size_t convolution_parameters = 2 * 2 * 3 * 3 + 2;
    for (std::size_t unit = 0; unit < 32; ++unit) {
        parameters[convolution_parameters + unit * 32 + unit] = 1.0;
    }
    identity.set_parameters(parameters);

    std::vector<double> image(32);
    for (std::size_t index = 0; index < image.size(); ++index) {
        image[index] = 0.1 * static_cast<double>(index) - 1.0;
    }
    const auto output = identity.forward(image);
    for (std::size_t index = 0; index < image.size(); ++index) {
        require_near(output[index], image[index], 1e-12,
                     "a zeroed residual convolution is not the identity");
    }

    // Gradient check through a residual block, which is the real verification.
    const ml_scratch::LabeledDataset images{
        {std::vector<double>(32, 0.25), 0},
        {std::vector<double>(32, -0.5), 1},
        {image, 0},
    };
    ml_scratch::ConvolutionalNetwork network{
        {2, 4, 4},
        {ml_scratch::residual_convolution(2, 3, ml_scratch::Activation::hyperbolic_tangent),
         ml_scratch::residual_convolution(2, 3, ml_scratch::Activation::hyperbolic_tangent),
         ml_scratch::max_pooling(2), ml_scratch::flatten(),
         ml_scratch::dense(2, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        13};

    const auto analytic = network.gradient(images);
    const auto original = network.parameters();
    std::vector<double> perturbed = original;
    std::vector<double> numerical(original.size(), 0.0);
    constexpr double epsilon = 1e-5;
    for (std::size_t index = 0; index < original.size(); ++index) {
        perturbed[index] = original[index] + epsilon;
        network.set_parameters(perturbed);
        const double raised = network.loss(images);
        perturbed[index] = original[index] - epsilon;
        network.set_parameters(perturbed);
        const double lowered = network.loss(images);
        perturbed[index] = original[index];
        numerical[index] = (raised - lowered) / (2.0 * epsilon);
    }
    network.set_parameters(original);
    require(ml_scratch::check_gradient(analytic, numerical).passed,
            "the residual convolution gradient failed its check");

    // The flag must survive a checkpoint, or a restored network would compute a different function.
    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_residual.checkpoint").string();
    network.save(path);
    const auto restored = ml_scratch::ConvolutionalNetwork::load(path);
    require(restored.layers() == network.layers(), "the residual flag did not survive");
    require(restored.parameters() == network.parameters(), "parameters did not survive");
    require_near(restored.loss(images), network.loss(images), 0.0,
                 "a restored residual network scored differently");
    std::filesystem::remove(path);
}

void test_dense_checkpoint_carries_the_skip() {
    const std::vector<ml_scratch::DenseLayer> layers{
        {4, 5, ml_scratch::Activation::hyperbolic_tangent},
        {5, 5, ml_scratch::Activation::rectified_linear, ml_scratch::Normalization::none, 0.0,
         true},
        {5, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::softmax_cross_entropy, 23};
    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_residual_dense.checkpoint").string();
    network.save(path);
    const auto restored = ml_scratch::FeedForwardNetwork::load(path);

    require(restored.layers() == network.layers(), "the residual flag did not survive");
    require(restored.layers()[1].residual, "the restored layer lost its skip");
    require_near(restored.loss(four_feature_data), network.loss(four_feature_data), 0.0,
                 "a restored residual network scored differently");
    std::filesystem::remove(path);

    require_invalid_argument(
        [] {
            ml_scratch::FeedForwardNetwork bad{{{4, 5, ml_scratch::Activation::identity,
                                                 ml_scratch::Normalization::none, 0.0, true}},
                                               ml_scratch::Loss::mean_squared_error};
        },
        "a residual layer accepted mismatched sizes");
}

// The CIFAR-10 reader is tested against a file the test writes, so it needs no download.
void test_cifar10_reader() {
    constexpr std::size_t record_size = 1 + 3 * 32 * 32;
    std::vector<unsigned char> bytes;
    for (std::size_t record = 0; record < 3; ++record) {
        bytes.push_back(static_cast<unsigned char>(record * 3)); // labels 0, 3, 6
        for (std::size_t pixel = 0; pixel < record_size - 1; ++pixel) {
            bytes.push_back(static_cast<unsigned char>((record * 7 + pixel) % 256));
        }
    }

    const auto write = [](const std::string& name, const std::vector<unsigned char>& data) {
        const std::string target = (std::filesystem::temp_directory_path() / name).string();
        std::ofstream stream{target, std::ios::binary};
        stream.write(reinterpret_cast<const char*>(data.data()),
                     static_cast<std::streamsize>(data.size()));
        return target;
    };
    const std::string path = write("ml_scratch_cifar.bin", bytes);

    const auto split = ml_scratch::load_cifar10_batch(path);
    require(split.samples.size() == 3, "wrong record count");
    require(split.channels == 3 && split.rows == 32 && split.columns == 32, "wrong shape");
    require(split.samples[0].label == 0 && split.samples[1].label == 3 &&
                split.samples[2].label == 6,
            "wrong labels");
    require(split.samples[0].features.size() == 3072, "wrong feature count");
    // The first record's pixels run 0, 1, 2, ... rescaled into [0, 1].
    require_near(split.samples[0].features[0], 0.0, 1e-12, "incorrect normalization");
    require_near(split.samples[0].features[1], 1.0 / 255.0, 1e-12, "incorrect normalization");
    require_near(split.samples[1].features[0], 7.0 / 255.0, 1e-12, "incorrect normalization");

    const auto capped = ml_scratch::load_cifar10_batch(path, 2);
    require(capped.samples.size() == 2, "max_samples was ignored");

    // Two files concatenate, and the cap applies across both.
    const auto combined = ml_scratch::load_cifar10({path, path});
    require(combined.samples.size() == 6, "batches did not concatenate");
    const auto limited = ml_scratch::load_cifar10({path, path}, 4);
    require(limited.samples.size() == 4, "the cap did not apply across batches");

    std::vector<unsigned char> truncated(bytes.begin(), bytes.begin() + record_size + 10);
    const std::string short_path = write("ml_scratch_cifar_short.bin", truncated);
    require_runtime_error([&] { static_cast<void>(ml_scratch::load_cifar10_batch(short_path)); },
                          "a truncated record was accepted");

    std::vector<unsigned char> bad_label = bytes;
    bad_label[0] = 42;
    const std::string bad_path = write("ml_scratch_cifar_label.bin", bad_label);
    require_runtime_error([&] { static_cast<void>(ml_scratch::load_cifar10_batch(bad_path)); },
                          "a label outside 0..9 was accepted");

    require_runtime_error(
        [] { static_cast<void>(ml_scratch::load_cifar10_batch("/no/such/file")); },
        "a missing file was accepted");
    require(ml_scratch::cifar10_class_names().size() == 10, "wrong class-name count");
    require(ml_scratch::cifar10_class_names()[0] == "airplane", "wrong class name");

    for (const std::string& target : {path, short_path, bad_path}) {
        std::filesystem::remove(target);
    }
}

void test_defaults_are_unchanged() {
    const ml_scratch::DenseLayer dense{2, 3, ml_scratch::Activation::identity};
    require(!dense.residual, "a dense layer carries a skip by default");
    const auto convolution = ml_scratch::convolution(4, 3, ml_scratch::Activation::identity);
    require(!convolution.residual, "a convolution carries a skip by default");
}

} // namespace

int main() {
    try {
        test_skip_is_added_before_the_activation();
        test_residual_gradients();
        test_skips_preserve_early_gradients();
        test_residual_convolution();
        test_dense_checkpoint_carries_the_skip();
        test_cifar10_reader();
        test_defaults_are_unchanged();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
